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
        x_test, y_test = f['x_test'], f['y_test']
    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test  = x_test.reshape(x_test.shape[0],  -1) / 255.0

    train_mask = (y_train == 4) | (y_train == 9)
    test_mask  = (y_test  == 4) | (y_test  == 9)

    x_train_f = x_train[train_mask]
    y_train_f = y_train[train_mask]
    x_test_f  = x_test[test_mask]
    y_test_f  = y_test[test_mask]

    y_train_f = np.where(y_train_f == 4, -1, 1)
    y_test_f  = np.where(y_test_f  == 4, -1, 1)

    idx_neg = np.where(y_train_f == -1)[0]
    idx_pos = np.where(y_train_f ==  1)[0]

    val_idx   = np.concatenate([idx_neg[:1000], idx_pos[:1000]])
    train_idx = np.concatenate([idx_neg[1000:], idx_pos[1000:]])

    x_val       = x_train_f[val_idx];   y_val       = y_train_f[val_idx]
    x_train_fin = x_train_f[train_idx]; y_train_fin = y_train_f[train_idx]

    return x_train_fin, y_train_fin, x_val, y_val, x_test_f, y_test_f

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


class DecisionStumpRegressor:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.c_left = None
        self.c_right = None

    def fit(self, X, y, iteration):
        n_samples, n_features = X.shape
        min_ssr = float('inf')

        for feature_i in range(n_features):
            X_col = X[:, feature_i]
            unique_vals = np.sort(np.unique(X_col))
            midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2

            if len(midpoints) > 1000:
                np.random.seed(iteration * n_features + feature_i)
                thresholds = np.random.choice(midpoints, 1000, replace=False)
            else:
                thresholds = midpoints

            left_mask = X_col[:, None] < thresholds[None, :]   # (N, T)
            N_L = np.sum(left_mask, axis=0)
            N_R = n_samples - N_L
            valid = (N_L > 0) & (N_R > 0)
            if not np.any(valid):
                continue

            Sum_L = np.sum(y[:, None] * left_mask, axis=0)
            Sum_R = np.sum(y) - Sum_L

            score = np.zeros(len(thresholds))
            score[valid] = (Sum_L[valid]**2 / N_L[valid]) + (Sum_R[valid]**2 / N_R[valid])

            best_idx = np.argmax(score)
            ssr = np.sum(y**2) - score[best_idx]

            if ssr < min_ssr:
                self.feature_index = feature_i
                self.threshold     = thresholds[best_idx]
                self.c_left        = Sum_L[best_idx] / N_L[best_idx]
                self.c_right       = Sum_R[best_idx] / N_R[best_idx]
                min_ssr = ssr

    def predict(self, X):
        col  = X[:, self.feature_index]
        preds = np.where(col < self.threshold, self.c_left, self.c_right)
        return preds

def train_gbm(X, y, X_val, y_val, n_clf=300, eta=0.01):
    F     = np.zeros(X.shape[0])
    F_val = np.zeros(X_val.shape[0])
    clfs, train_mses, val_mses = [], [], []

    for i in range(n_clf):
        r = np.sign(y - F)  # negative gradient of absolute loss

        clf = DecisionStumpRegressor()
        clf.fit(X, r, iteration=i)

        F     += eta * clf.predict(X)
        F_val += eta * clf.predict(X_val)

        train_mses.append(np.mean((y     - F    )**2))
        val_mses.append(  np.mean((y_val - F_val)**2))
        clfs.append(clf)

    return clfs, train_mses, val_mses

def predict_gbm(X, clfs, eta, max_idx):
    F = np.zeros(X.shape[0])
    for clf in clfs[:max_idx]:
        F += eta * clf.predict(X)
    return F


def main():
    x_train, y_train, x_test, y_test = load_mnist()
    x_tr, y_tr, x_val, y_val, x_te, y_te = preprocess_data(
        x_train, y_train, x_test, y_test)
    x_tr_p, x_val_p, x_te_p = apply_pca(x_tr, x_val, x_te, n_components=5)

    clfs_01, tr_mse_01, val_mse_01 = train_gbm(
        x_tr_p, y_tr, x_val_p, y_val, n_clf=300, eta=0.01)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 301), val_mse_01,  label='Validation MSE', color='red',  linewidth=2)
    plt.plot(range(1, 301), tr_mse_01,   label='Train MSE',      color='blue', linewidth=2)
    plt.title('Train & Validation MSE vs Number of Trees  (η = 0.01)')
    plt.xlabel('Number of Trees'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True)
    plt.savefig('q2_mse_eta_0.01.png', dpi=150, bbox_inches='tight')
    plt.close()

    best_01   = np.argmin(val_mse_01)
    te_preds  = predict_gbm(x_te_p, clfs_01, eta=0.01, max_idx=best_01 + 1)
    te_mse_01 = np.mean((y_te - te_preds)**2)

    print(f"eta=0.01 | Best Iter: {best_01+1:>3} | "
          f"Train MSE: {tr_mse_01[best_01]:.4f} | "
          f"Val MSE: {val_mse_01[best_01]:.4f} | "
          f"Test MSE: {te_mse_01:.4f}")

    learning_rates = [0.001, 0.01, 0.1, 0.2, 0.5, 1]
    plt.figure(figsize=(12, 8))
    print(f"\n{'eta':<8} {'Best Iter':<12} {'Train MSE':<12} {'Val MSE':<12} {'Test MSE'}")
    print("-" * 58)

    for eta in learning_rates:
        if eta == 0.01:
            clfs, tr_mses, val_mses = clfs_01, tr_mse_01, val_mse_01
        else:
            clfs, tr_mses, val_mses = train_gbm(
                x_tr_p, y_tr, x_val_p, y_val, n_clf=300, eta=eta)

        plt.plot(range(1, 301), val_mses, label=f'η={eta}')

        best_i = np.argmin(val_mses)
        te_p   = predict_gbm(x_te_p, clfs, eta=eta, max_idx=best_i + 1)
        te_mse = np.mean((y_te - te_p)**2)

        print(f"{eta:<8} {best_i+1:<12} {tr_mses[best_i]:<12.4f} "
              f"{val_mses[best_i]:<12.4f} {te_mse:.4f}")

    plt.title('Validation MSE vs Number of Trees — all η')
    plt.xlabel('Number of Trees'); plt.ylabel('Validation MSE')
    plt.legend(); plt.grid(True)
    plt.savefig('q2_mse_all_etas.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()