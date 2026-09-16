
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


# 1. Load and preprocess MNIST data
data = np.load("mnist.npz")
X_train_full, y_train_full = data["x_train"], data["y_train"]
X_test_full,  y_test_full  = data["x_test"],  data["y_test"]

# Select classes 0, 1, 2
def filter_classes(X, y, classes=(0, 1, 2)):
    mask = np.isin(y, classes)
    return X[mask], y[mask]

X_train_raw, y_train = filter_classes(X_train_full, y_train_full)
X_test_raw,  y_test  = filter_classes(X_test_full,  y_test_full)

# Flatten and normalize
X_train_flat = X_train_raw.reshape(len(X_train_raw), -1).astype(np.float64) / 255.0
X_test_flat  = X_test_raw.reshape(len(X_test_raw),  -1).astype(np.float64) / 255.0


# 2. PCA – reduce to p = 10 using full training set
p = 10

# Center using training mean
mean_train = X_train_flat.mean(axis=0)
X_train_c  = X_train_flat - mean_train
X_test_c   = X_test_flat  - mean_train

# SVD on centered training data
U, S, Vt = np.linalg.svd(X_train_c, full_matrices=False)
W_pca = Vt[:p]                                # (p, 784)

X_train_pca = X_train_c @ W_pca.T            # (N_train, p)
X_test_pca  = X_test_c  @ W_pca.T            # (N_test,  p)


# 3. One-vs-rest target matrices
classes = [0, 1, 2]
K = len(classes)

def make_target_matrix(y, classes):
    """Returns Y of shape (N, K) with 1-hot one-vs-rest encoding."""
    Y = np.zeros((len(y), len(classes)))
    for i, c in enumerate(classes):
        Y[:, i] = (y == c).astype(float)
    return Y

Y_train = make_target_matrix(y_train, classes)   # (N_train, 3)
Y_test  = make_target_matrix(y_test,  classes)   # (N_test,  3)


# 4. Ridge Regression (closed-form)

lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

def ridge_fit(X, Y, lam):
    """W = (X^T X + λI)^{-1} X^T Y  →  shape (p, K)"""
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ Y)

def mse(X, Y, W):
    pred = X @ W
    return np.mean((Y - pred) ** 2)

def predict_class(X, W):
    """Argmax over K outputs."""
    scores = X @ W
    return np.argmax(scores, axis=1)

ridge_train_mse = []
ridge_test_mse  = []
ridge_W_list    = []

for lam in lambdas:
    W = ridge_fit(X_train_pca, Y_train, lam)
    ridge_W_list.append(W)
    ridge_train_mse.append(mse(X_train_pca, Y_train, W))
    ridge_test_mse.append(mse(X_test_pca,  Y_test,  W))

print("\n--- Ridge Regression ---")
print(f"{'Lambda':>10}  {'Train MSE':>12}  {'Test MSE':>12}")
for lam, tr, te in zip(lambdas, ridge_train_mse, ridge_test_mse):
    print(f"{lam:>10.4f}  {tr:>12.6f}  {te:>12.6f}")


# 5. Lasso Regression (sklearn only for Lasso)

lasso_train_mse   = []
lasso_test_mse    = []
lasso_nonzero     = []
lasso_W_list      = []

for lam in lambdas:
    W_cols = []
    for k in range(K):
        model = Lasso(alpha=lam, max_iter=10000, tol=1e-4)
        model.fit(X_train_pca, Y_train[:, k])
        W_cols.append(model.coef_)
    W = np.column_stack(W_cols)   # (p, K)
    lasso_W_list.append(W)

    lasso_train_mse.append(mse(X_train_pca, Y_train, W))
    lasso_test_mse.append(mse(X_test_pca,  Y_test,  W))
    lasso_nonzero.append(int(np.sum(np.abs(W) > 1e-8)))

print("\n--- Lasso Regression ---")
print(f"{'Lambda':>10}  {'Train MSE':>12}  {'Test MSE':>12}  {'Non-zero coefs':>15}")
for lam, tr, te, nz in zip(lambdas, lasso_train_mse, lasso_test_mse, lasso_nonzero):
    print(f"{lam:>10.4f}  {tr:>12.6f}  {te:>12.6f}  {nz:>15}")


# 6. Plot: Train/Test MSE vs λ (Ridge & Lasso)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(lambdas, ridge_train_mse, 'bo-', label="Train MSE")
axes[0].plot(lambdas, ridge_test_mse,  'rs-', label="Test MSE")
axes[0].set_xscale("log")
axes[0].set_xlabel("λ (log scale)")
axes[0].set_ylabel("MSE")
axes[0].set_title("Ridge Regression: MSE vs λ")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(lambdas, lasso_train_mse, 'bo-', label="Train MSE")
axes[1].plot(lambdas, lasso_test_mse,  'rs-', label="Test MSE")
axes[1].set_xscale("log")
axes[1].set_xlabel("λ (log scale)")
axes[1].set_ylabel("MSE")
axes[1].set_title("Lasso Regression: MSE vs λ")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("q1_mse_vs_lambda.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 7. Plot: Non-zero coefficients vs λ (Lasso)
# ──────────────────────────────────────────────
plt.figure(figsize=(7, 5))
plt.plot(lambdas, lasso_nonzero, 'go-')
plt.xscale("log")
plt.xlabel("λ (log scale)")
plt.ylabel("Number of non-zero coefficients")
plt.title("Lasso: Non-zero Coefficients vs λ")
plt.grid(True)
plt.tight_layout()
plt.savefig("q1_lasso_nonzero.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 8. Regularization paths – Ridge (class 1)
# ──────────────────────────────────────────────
# ridge_W_list[i] shape: (p, K) — column 1 is class 1
ridge_coefs_class1 = np.array([W[:, 1] for W in ridge_W_list])  # (len(lambdas), p)

plt.figure(figsize=(8, 5))
for feat in range(p):
    plt.plot(lambdas, ridge_coefs_class1[:, feat], marker='o', label=f"w{feat+1}")
plt.xscale("log")
plt.xlabel("λ (log scale)")
plt.ylabel("Coefficient value")
plt.title("Ridge Regularization Path (Class 1)")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("q1_ridge_reg_path.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 9. Regularization paths – Lasso (class 1)
# ──────────────────────────────────────────────
lasso_coefs_class1 = np.array([W[:, 1] for W in lasso_W_list])  # (len(lambdas), p)

plt.figure(figsize=(8, 5))
for feat in range(p):
    plt.plot(lambdas, lasso_coefs_class1[:, feat], marker='o', label=f"w{feat+1}")
plt.xscale("log")
plt.xlabel("λ (log scale)")
plt.ylabel("Coefficient value")
plt.title("Lasso Regularization Path (Class 1)")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("q1_lasso_reg_path.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 10. Vary model complexity (p = 2, 5, 10, 20, 30)
#     Use best λ from Ridge (lowest test MSE)
# ──────────────────────────────────────────────
best_lambda_ridge = lambdas[np.argmin(ridge_test_mse)]

p_values = [2, 5, 10, 20, 30]
complexity_train_mse = []
complexity_test_mse  = []

for pv in p_values:
    # Project to pv dimensions using the same PCA basis
    W_pv = Vt[:pv]
    Xtr_pv = X_train_c @ W_pv.T
    Xte_pv = X_test_c  @ W_pv.T

    W_r = ridge_fit(Xtr_pv, Y_train, best_lambda_ridge)
    complexity_train_mse.append(mse(Xtr_pv, Y_train, W_r))
    complexity_test_mse.append(mse(Xte_pv,  Y_test,  W_r))

print("\n--- Ridge: MSE vs Model Complexity ---")
print(f"{'p':>5}  {'Train MSE':>12}  {'Test MSE':>12}")
for pv, tr, te in zip(p_values, complexity_train_mse, complexity_test_mse):
    print(f"{pv:>5}  {tr:>12.6f}  {te:>12.6f}")

plt.figure(figsize=(7, 5))
plt.plot(p_values, complexity_train_mse, 'bo-', label="Train MSE")
plt.plot(p_values, complexity_test_mse,  'rs-', label="Test MSE")
plt.xlabel("Model Complexity (p = PCA dimensions)")
plt.ylabel("MSE")
plt.title(f"Ridge Regression: MSE vs PCA Dimensions (λ={best_lambda_ridge})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("q1_complexity.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 11. Test classification accuracy for best Ridge and Lasso models
# ──────────────────────────────────────────────
# Best Ridge model (lowest test MSE at p=10)
best_ridge_idx = np.argmin(ridge_test_mse)
best_ridge_W   = ridge_W_list[best_ridge_idx]
best_ridge_lam = lambdas[best_ridge_idx]

ridge_preds = predict_class(X_test_pca, best_ridge_W)
# y_test contains original labels 0, 1, 2 — map predictions back
ridge_acc = np.mean(ridge_preds == y_test)
print(f"\n--- Test Classification Accuracy ---")
print(f"Best Ridge λ = {best_ridge_lam}  →  Test Accuracy = {ridge_acc*100:.2f}%")

# Best Lasso model (lowest test MSE at p=10)
best_lasso_idx = np.argmin(lasso_test_mse)
best_lasso_W   = lasso_W_list[best_lasso_idx]
best_lasso_lam = lambdas[best_lasso_idx]

lasso_preds = predict_class(X_test_pca, best_lasso_W)
lasso_acc = np.mean(lasso_preds == y_test)
print(f"Best Lasso λ = {best_lasso_lam}  →  Test Accuracy = {lasso_acc*100:.2f}%")