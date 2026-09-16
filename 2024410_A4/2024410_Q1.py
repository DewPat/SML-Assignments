import numpy as np
import urllib.request
import os
import matplotlib.pyplot as plt

def load_mnist():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    filename = "mnist.npz"
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    with np.load(filename, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test   = f['x_test'],  f['y_test']
    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test  = x_test.reshape(x_test.shape[0],  -1) / 255.0

    train_mask = (y_train == 4) | (y_train == 9)
    test_mask  = (y_test  == 4) | (y_test  == 9)

    x_train_f = x_train[train_mask];  y_train_f = y_train[train_mask]
    x_test_f  = x_test[test_mask];    y_test_f  = y_test[test_mask]

    y_train_f = np.where(y_train_f == 4, -1, 1)
    y_test_f  = np.where(y_test_f  == 4, -1, 1)

    idx_neg = np.where(y_train_f == -1)[0]
    idx_pos = np.where(y_train_f ==  1)[0]

    val_idx   = np.concatenate([idx_neg[:1000], idx_pos[:1000]])
    train_idx = np.concatenate([idx_neg[1000:], idx_pos[1000:]])

    return (x_train_f[train_idx], y_train_f[train_idx],
            x_train_f[val_idx],   y_train_f[val_idx],
            x_test_f,             y_test_f)

def apply_pca(x_train, x_val, x_test, n_components=5):
    mean = np.mean(x_train, axis=0)
    x_train_c = x_train - mean
    x_val_c   = x_val   - mean
    x_test_c  = x_test  - mean

    cov = np.cov(x_train_c, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx][:, :n_components]

    return (np.dot(x_train_c, components),
            np.dot(x_val_c,   components),
            np.dot(x_test_c,  components))

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold     = None
        self.polarity      = 1
        self.alpha         = None

    def predict(self, X):
        X_col = X[:, self.feature_index]
        preds = np.ones(X.shape[0])
        if self.polarity == 1:
            preds[X_col < self.threshold] = -1
        else:
            preds[X_col >= self.threshold] = -1
        return preds

def train_adaboost(X, y, X_val, y_val, n_clf=300):
    n_samples, n_features = X.shape
    w = np.full(n_samples, 1 / n_samples)

    clfs           = []
    val_accuracies = []
    EPS            = 1e-10

    for i in range(n_clf):
        clf       = DecisionStump()
        min_error = float('inf')

        for feature_i in range(n_features):
            X_col      = X[:, feature_i]
            unique_vals = np.sort(np.unique(X_col))
            midpoints  = (unique_vals[:-1] + unique_vals[1:]) / 2

            if len(midpoints) > 1000:
                np.random.seed(i * n_features + feature_i)
                thresholds = np.random.choice(midpoints, 1000, replace=False)
            else:
                thresholds = midpoints

            left_mask = X_col[:, None] < thresholds[None, :] 

            preds_pos = np.ones((n_samples, len(thresholds)))
            preds_pos[left_mask] = -1

            err_pos = np.sum(w[:, None] * (y[:, None] != preds_pos),  axis=0)
            err_neg = np.sum(w[:, None] * (y[:, None] != -preds_pos), axis=0)

            min_pos_idx = np.argmin(err_pos)
            min_neg_idx = np.argmin(err_neg)

            if err_pos[min_pos_idx] < min_error:
                clf.polarity      = 1
                clf.threshold     = thresholds[min_pos_idx]
                clf.feature_index = feature_i
                min_error         = err_pos[min_pos_idx]

            if err_neg[min_neg_idx] < min_error:
                clf.polarity      = -1
                clf.threshold     = thresholds[min_neg_idx]
                clf.feature_index = feature_i
                min_error         = err_neg[min_neg_idx]

        clf.alpha = 0.5 * np.log((1 - min_error + EPS) / (min_error + EPS))

        predictions = clf.predict(X)
        w *= np.exp(-clf.alpha * y * predictions)
        w /= np.sum(w)

        clfs.append(clf)

        # validation accuracy after adding this stump
        val_preds = np.sign(sum(c.alpha * c.predict(X_val) for c in clfs))
        val_accuracies.append(np.mean(val_preds == y_val))

    return clfs, val_accuracies

def predict_adaboost(X, clfs):
    return np.sign(sum(c.alpha * c.predict(X) for c in clfs))

def main():
    x_train, y_train, x_test, y_test = load_mnist()
    x_tr, y_tr, x_val, y_val, x_te, y_te = preprocess_data(
        x_train, y_train, x_test, y_test)
    x_tr_p, x_val_p, x_te_p = apply_pca(x_tr, x_val, x_te, n_components=5)

    clfs, val_accuracies = train_adaboost(
        x_tr_p, y_tr, x_val_p, y_val, n_clf=300)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 301), val_accuracies, color='b', linewidth=2)
    plt.title('Validation Accuracy vs Number of Stumps (AdaBoost)')
    plt.xlabel('Number of Stumps')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.savefig('q1_val_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()

    test_accuracies      = []
    test_preds_running   = np.zeros(x_te_p.shape[0])
    for c in clfs:
        test_preds_running += c.alpha * c.predict(x_te_p)
        test_accuracies.append(np.mean(np.sign(test_preds_running) == y_te))

    print(f"\n{'Trees':<10} {'Val Acc':<12} {'Test Acc'}")
    print("-" * 34)
    for trees in [50, 100, 150, 200, 250, 300]:
        idx = trees - 1
        print(f"{trees:<10} {val_accuracies[idx]:<12.4f} {test_accuracies[idx]:.4f}")

    best_iter = np.argmax(val_accuracies)
    print(f"\nBest Iter: {best_iter+1} | "
          f"Val Acc: {val_accuracies[best_iter]:.4f} | "
          f"Test Acc: {test_accuracies[best_iter]:.4f}")

if __name__ == "__main__":
    main()