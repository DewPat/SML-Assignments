import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import os
import urllib.request


# data helpers
def get_data():
    url  = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist.npz")
    if not os.path.exists(path):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    data = np.load(path)
    return data['x_train'], data['y_train'], data['x_test'], data['y_test']


def prep(xtr, ytr, xte, yte, cls=(0, 1, 2)):
    tr_idx = np.isin(ytr, cls)
    te_idx = np.isin(yte, cls)
    xtr, ytr = xtr[tr_idx], ytr[tr_idx]
    xte, yte = xte[te_idx], yte[te_idx]
    xtr_f = xtr.reshape(xtr.shape[0], -1) / 255.0
    xte_f = xte.reshape(xte.shape[0], -1) / 255.0
    return xtr_f, ytr, xte_f, yte


# PCA  (fit on train, project both splits)
def PCA(xtr, xte, p=10):
    mu    = np.mean(xtr, axis=0)
    xtr_c = xtr - mu
    xte_c = xte - mu
    cov   = np.cov(xtr_c, rowvar=False)
    val, vec = np.linalg.eigh(cov)
    idx   = np.argsort(val)[::-1]
    top   = vec[:, idx][:, :p]
    return xtr_c @ top, xte_c @ top


# Gini impurity
def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.array([np.sum(y == k) / len(y) for k in range(3)])
    return 1.0 - float(np.sum(p ** 2))

# best split over given features
def best_split(X, y, feat_ids):
    best_g, result = float('inf'), None
    n = len(y)
    for f in feat_ids:
        thresh = np.mean(X[:, f])
        lm = X[:, f] <= thresh
        rm = ~lm
        if lm.sum() == 0 or rm.sum() == 0:
            continue
        gw = (lm.sum() / n) * gini(y[lm]) + (rm.sum() / n) * gini(y[rm])
        if gw < best_g:
            best_g = gw
            result = (f, thresh, lm, rm, gw)
    return result

# majority-class label
def majority(y):
    if len(y) == 0:
        return 0
    vals, counts = np.unique(y, return_counts=True)
    return int(vals[np.argmax(counts)])


# Build tree  (greedy best-first leaf expansion)
# n_leaves : stop when we have this many leaf nodes
# k        : random feature subset size (None = all features)
def build_tree(X, y, n_leaves=3, k=None):
    root   = dict(leaf=True, label=majority(y),
                  mask=np.ones(len(y), dtype=bool))
    leaves = [root]

    while len(leaves) < n_leaves:
        best_gain = float('inf')
        best_leaf = None
        best_info = None

        for node in leaves:
            Xl = X[node['mask']]
            yl = y[node['mask']]
            if len(np.unique(yl)) <= 1:
                continue
            feat_ids = (np.random.choice(X.shape[1], k, replace=False).tolist()
                        if k is not None else list(range(X.shape[1])))
            res = best_split(Xl, yl, feat_ids)
            if res is None:
                continue
            f, th, lm_loc, rm_loc, gw = res
            if gw < best_gain:
                best_gain = gw
                best_leaf = node
                gi = np.where(node['mask'])[0]
                glm = np.zeros(len(y), dtype=bool)
                grm = np.zeros(len(y), dtype=bool)
                glm[gi[lm_loc]] = True
                grm[gi[rm_loc]] = True
                best_info = (f, th, glm, grm)

        if best_leaf is None:
            break

        f, th, glm, grm = best_info
        left  = dict(leaf=True, label=majority(y[glm]), mask=glm)
        right = dict(leaf=True, label=majority(y[grm]), mask=grm)
        best_leaf.update(dict(leaf=False, feat=f, th=th, left=left, right=right))
        best_leaf.pop('label', None)
        leaves.remove(best_leaf)
        leaves.extend([left, right])

    return root

# predict
def predict_one(x, node):
    while not node['leaf']:
        node = node['left'] if x[node['feat']] <= node['th'] else node['right']
    return node['label']


def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])


# print tree structure
def print_tree(node, y_train, depth=0):
    pad = "  " * (depth + 2)
    if node['leaf']:
        msk  = node['mask']
        dist = {c: int(np.sum(y_train[msk] == c)) for c in (0, 1, 2)}
        print(f"{pad}[LEAF] label={node['label']}  dist={dist}  n={msk.sum()}")
    else:
        print(f"{pad}[SPLIT] feat={node['feat']}  thresh={node['th']:.4f}")
        print(f"{pad}  LEFT  (feat <= thresh):")
        print_tree(node['left'],  y_train, depth + 2)
        print(f"{pad}  RIGHT (feat >  thresh):")
        print_tree(node['right'], y_train, depth + 2)

# accuracy report
def report_accuracy(preds, labels, title=""):
    acc = float(np.mean(preds == labels))
    print(f"\n  {title}")
    print(f"  Overall accuracy : {acc*100:.2f}%")
    print(f"  {'Class':>6}  {'Correct':>9}  {'Total':>7}  {'Acc':>8}")
    for c in (0, 1, 2):
        mask = labels == c
        ca   = float(np.mean(preds[mask] == labels[mask]))
        print(f"  {c:>6}  {int(ca*mask.sum()):>9}  {int(mask.sum()):>7}  {ca*100:>7.2f}%")
    return acc


# majority vote ensemble
def ensemble_vote(preds_matrix):
    N = preds_matrix.shape[0]
    out = np.zeros(N, dtype=int)
    for i in range(N):
        vals, counts = np.unique(preds_matrix[i], return_counts=True)
        out[i] = int(vals[np.argmax(counts)])
    return out


# main
def main():
    np.random.seed(42)
    N_TREES  = 5
    N_LEAVES = 3
    K_FOREST = 3      

    SEP = "=" * 65
    sep = "-" * 65

    xtr_raw, ytr_raw, xte_raw, yte_raw = get_data()
    xtr, ytr, xte, yte = prep(xtr_raw, ytr_raw, xte_raw, yte_raw)

    P = 10
    xp_tr, xp_te = PCA(xtr, xte, p=P)

    # PART 1 - Single Decision Tree
    tree    = build_tree(xp_tr, ytr, n_leaves=N_LEAVES)
    pred_te = predict(xp_te, tree)
    report_accuracy(pred_te, yte, title="Single Decision Tree - Test Set")

    # PART 2 - Bagging
    bag_trees  = []
    bag_oob    = []
    n_tr = len(ytr)

    for b in range(N_TREES):
        idx  = np.random.choice(n_tr, n_tr, replace=True)
        oidx = np.setdiff1d(np.arange(n_tr), idx)
        t    = build_tree(xp_tr[idx], ytr[idx], n_leaves=N_LEAVES)
        bag_trees.append(t)

        oob_e = float(np.mean(predict(xp_tr[oidx], t) != ytr[oidx])) if len(oidx) > 0 else float('nan')
        bag_oob.append(oob_e)
        print(f"  Tree {b+1}: bootstrap={len(idx)}  OOB={len(oidx)}  OOB-error={oob_e:.4f}")

    avg_oob = float(np.nanmean(bag_oob))
    print(f"\n  Average OOB Error (Bagging) : {avg_oob:.4f}  ({avg_oob*100:.2f}%)")

    bag_preds = np.column_stack([predict(xp_te, t) for t in bag_trees])
    final_bag = ensemble_vote(bag_preds)
    report_accuracy(final_bag, yte, title="Bagging - Test Set (majority vote)")

    # PART 3 - Random Forest
    print(f"  k choice: k = floor(sqrt(p))= {K_FOREST}")

    np.random.seed(42)   # reset so bootstrap samples match Bagging
    rf_trees = []
    rf_oob   = []

    for b in range(N_TREES):
        idx  = np.random.choice(n_tr, n_tr, replace=True)
        oidx = np.setdiff1d(np.arange(n_tr), idx)
        t    = build_tree(xp_tr[idx], ytr[idx], n_leaves=N_LEAVES, k=K_FOREST)
        rf_trees.append(t)

        oob_e = float(np.mean(predict(xp_tr[oidx], t) != ytr[oidx])) if len(oidx) > 0 else float('nan')
        rf_oob.append(oob_e)
        print(f"  Tree {b+1}: bootstrap={len(idx)}  OOB={len(oidx)}  OOB-error={oob_e:.4f}")

    avg_rf_oob = float(np.nanmean(rf_oob))
    print(f"\n  Average OOB Error (Random Forest) : {avg_rf_oob:.4f}  ({avg_rf_oob*100:.2f}%)")

    rf_preds = np.column_stack([predict(xp_te, t) for t in rf_trees])
    final_rf = ensemble_vote(rf_preds)
    report_accuracy(final_rf, yte, title="Random Forest - Test Set (majority vote)")

    # -- Summary ------------------------------------------
    single_acc = float(np.mean(pred_te  == yte))
    bag_acc    = float(np.mean(final_bag == yte))
    rf_acc     = float(np.mean(final_rf  == yte))

    print("\n" + SEP)
    print("  SUMMARY")
    print(SEP)
    print(f"  {'Method':<30}  {'Test Acc':>10}  {'Avg OOB Err':>12}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*12}")
    print(f"  {'Single Decision Tree':<30}  {single_acc*100:>9.2f}%  {'N/A':>12}")
    print(f"  {'Bagging (5 trees)':<30}  {bag_acc*100:>9.2f}%  {avg_oob*100:>11.2f}%")
    print(f"  {'Random Forest (5 trees, k=3)':<30}  {rf_acc*100:>9.2f}%  {avg_rf_oob*100:>11.2f}%")

    better = "Random Forest" if rf_acc >= bag_acc else "Bagging"
    print(f"\n  Better ensemble: {better}")
    print(SEP)
    sys.stdout.flush()


if __name__ == '__main__':
    main()