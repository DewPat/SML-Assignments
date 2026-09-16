import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import urllib.request

np.random.seed(42)

# ──────────────────────────────────────────────
# data loading
# ──────────────────────────────────────────────
def load_fashion_mnist():
    import pandas as pd
    train = pd.read_csv("fashion-mnist_train.csv")
    test  = pd.read_csv("fashion-mnist_test.csv")

    ytr = train['label'].to_numpy()
    xtr = train.drop('label', axis=1).to_numpy().reshape(-1, 28, 28)

    yte = test['label'].to_numpy()
    xte = test.drop('label', axis=1).to_numpy().reshape(-1, 28, 28)

    return xtr, ytr, xte, yte

def filter_and_preprocess(xtr, ytr, xte, yte, classes=(0,1,2)):
    mtr = np.isin(ytr, classes)
    mte = np.isin(yte, classes)
    xtr, ytr = xtr[mtr].reshape(mtr.sum(), -1) / 255.0, ytr[mtr].astype(float)
    xte, yte = xte[mte].reshape(mte.sum(), -1) / 255.0, yte[mte].astype(float)
    return xtr, ytr, xte, yte

# ──────────────────────────────────────────────
# pca using svd
# ──────────────────────────────────────────────
def pca(xtr, xte, p=10):
    mu    = xtr.mean(axis=0)
    xtr_c = xtr - mu
    xte_c = xte - mu
    _, _, Vt = np.linalg.svd(xtr_c, full_matrices=False)
    W = Vt[:p]
    return xtr_c @ W.T, xte_c @ W.T

# ──────────────────────────────────────────────
# regression decision stump: minimise ssr
# thresholds = midpoints between consecutive sorted unique values
# ──────────────────────────────────────────────
def fit_stump(X, y):
    best_ssr, best = np.inf, None
    n = len(y)

    for f in range(X.shape[1]):
        order = np.argsort(X[:, f])
        xs, ys = X[order, f], y[order]

        t_sum = ys.sum()
        t_sq  = (ys**2).sum()
        sl = sql = 0.0
        nl = 0

        for i in range(n - 1):
            sl  += ys[i];  sql += ys[i]**2;  nl += 1
            if xs[i] == xs[i+1]:
                continue
            nr = n - nl
            sr = t_sum - sl
            ssr = (sql - sl**2/nl) + ((t_sq - sql) - sr**2/nr)
            if ssr < best_ssr:
                best_ssr = ssr
                best = (f, (xs[i]+xs[i+1])/2.0, sl/nl, sr/nr, ssr)

    return best  # (feat, threshold, mean_left, mean_right, ssr)

def predict_stump(X, stump):
    f, th, ml, mr, _ = stump
    return np.where(X[:, f] <= th, ml, mr)

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred)**2))

# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
xtr_raw, ytr_raw, xte_raw, yte_raw = load_fashion_mnist()
xtr, ytr, xte, yte = filter_and_preprocess(xtr_raw, ytr_raw, xte_raw, yte_raw)
xtr_p, xte_p = pca(xtr, xte, p=10)

# ── part 1: single stump h1 ──
s1 = fit_stump(xtr_p, ytr)
f1, th1, ml1, mr1, ssr1 = s1
p_single  = predict_stump(xte_p, s1)
mse_single = mse(yte, p_single)

print(f"[Part 1] Best split  : feature={f1}, threshold={th1:.6f}")
print(f"[Part 1] Left  mean  : {ml1:.6f}  (n={(xtr_p[:,f1]<=th1).sum()})")
print(f"[Part 1] Right mean  : {mr1:.6f}  (n={(xtr_p[:,f1]> th1).sum()})")
print(f"[Part 1] Train SSR   : {ssr1:.4f}")
print(f"[Part 1] Test MSE    : {mse_single:.6f}")

# ── part 2: bagging with 5 bootstrap stumps ──
N_TREES  = 5
n        = len(ytr)
stumps   = []
oob_mses = []

for b in range(N_TREES):
    idx  = np.random.choice(n, n, replace=True)
    oidx = np.setdiff1d(np.arange(n), idx)
    s    = fit_stump(xtr_p[idx], ytr[idx])
    stumps.append(s)
    oob_mse = mse(ytr[oidx], predict_stump(xtr_p[oidx], s)) if len(oidx) > 0 else np.nan
    oob_mses.append(oob_mse)
    print(f"[Bag {b+1}] feat={s[0]}, thresh={s[1]:.4f}, "
          f"OOB size={len(oidx)}, OOB MSE={oob_mse:.6f}")

p_bag     = np.column_stack([predict_stump(xte_p, s) for s in stumps]).mean(axis=1)
mse_bag   = mse(yte, p_bag)
avg_oob   = float(np.nanmean(oob_mses))

print(f"\n[Bagging] Avg OOB MSE : {avg_oob:.6f}")
print(f"[Bagging] Test MSE    : {mse_bag:.6f}")
print(f"\n{'Method':<28} {'Test MSE':>12} {'Avg OOB MSE':>14}")
print(f"{'-'*28} {'-'*12} {'-'*14}")
print(f"{'Single Decision Stump':<28} {mse_single:>12.6f} {'N/A':>14}")
print(f"{'Bagging (5 stumps)':<28} {mse_bag:>12.6f} {avg_oob:>14.6f}")

# ── plot: true vs single stump vs bagged predictions ──
sidx = np.argsort(yte)
sub  = np.arange(0, len(sidx), 30)
x_ax = np.arange(len(sub))

plt.figure(figsize=(12, 5))
plt.scatter(x_ax, yte[sidx][sub],      marker='|', s=120, color='royalblue',  lw=1.5, label='True label (y)',            zorder=3)
plt.scatter(x_ax, p_single[sidx][sub], marker='x', s=60,  color='darkorange', lw=1.5, label='Single stump h1(x)',        zorder=2, alpha=0.85)
plt.scatter(x_ax, p_bag[sidx][sub],    marker='^', s=40,  color='seagreen',           label='Bagged prediction',         zorder=1, alpha=0.75)
for v in [0, 1, 2]:
    plt.axhline(v, color='grey', lw=0.6, ls='--', alpha=0.4)
plt.yticks([0,1,2], ['0 (T-shirt)', '1 (Trouser)', '2 (Pullover)'])
plt.xlabel('Sorted sample index (every 30th test sample)')
plt.ylabel('Response value')
plt.title('Single Stump vs Bagging — Fashion-MNIST classes 0,1,2 (sorted by true label)')
plt.legend()
plt.grid(True, axis='y', ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig('q3_stump_bagging.png', dpi=150)