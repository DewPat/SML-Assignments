import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
mu_neg = np.array([-3, -3])   
mu_pos = np.array([ 3,  3]) 

n_per_class = 200

cov_A = np.eye(2)
X_neg_A = np.random.multivariate_normal(mu_neg, cov_A, n_per_class)
X_pos_A = np.random.multivariate_normal(mu_pos, cov_A, n_per_class)
X_A = np.vstack([X_neg_A, X_pos_A])
y_A = np.hstack([np.full(n_per_class, -1), np.ones(n_per_class)])

cov_B = 3 * np.eye(2)
X_neg_B = np.random.multivariate_normal(mu_neg, cov_B, n_per_class)
X_pos_B = np.random.multivariate_normal(mu_pos, cov_B, n_per_class)
X_B = np.vstack([X_neg_B, X_pos_B])
y_B = np.hstack([np.full(n_per_class, -1), np.ones(n_per_class)])


def train_test_split_manual(X, y, test_ratio=0.30, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    split = int(len(y) * (1 - test_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

X_trA, X_teA, y_trA, y_teA = train_test_split_manual(X_A, y_A)
X_trB, X_teB, y_trB, y_teB = train_test_split_manual(X_B, y_B)

def rosenblatt_perceptron(X_train, y_train, lr=0.01, max_epochs=300):
    n, d = X_train.shape
    w = np.zeros(d)        # initialise weights to zero
    b = 0.0                # initialise bias to zero
    misclass_per_epoch = []
    conv_epoch = None
    rng = np.random.default_rng(0)
    for epoch in range(1, max_epochs + 1):
        n_wrong = 0
        indices = rng.permutation(n)
        for i in indices:
            pred = np.sign(w @ X_train[i] + b)
            if pred == 0:
                pred = 1          # tie-break
            if pred != y_train[i]:
                n_wrong += 1
                # gradient-descent update
                w += lr * y_train[i] * X_train[i]
                b += lr * y_train[i]
        misclass_per_epoch.append(n_wrong)

        # early stopping
        if n_wrong == 0:
            conv_epoch = epoch
            print(f"  Converged at epoch {epoch}.")
            break
    if conv_epoch is None:
        print(f"  Did NOT converge within {max_epochs} epochs.")
    return w, b, misclass_per_epoch, conv_epoch


def accuracy(X, y, w, b):
    preds = np.sign(X @ w + b)
    preds[preds == 0] = 1
    return np.mean(preds == y) * 100

def plot_results(X_train, y_train, X_test, y_test,
                 w, b, misclass, conv_epoch,
                 dataset_name):
    epochs = np.arange(1, len(misclass) + 1)

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, misclass, color='#E63946', linewidth=2)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("# Misclassified", fontsize=11)
    plt.title(f"Dataset {dataset_name} – Misclassifications per Epoch", fontsize=12)
    plt.grid(True, alpha=0.3)
    if conv_epoch:
        plt.axvline(conv_epoch, color='#2A9D8F', linestyle='--',
                       label=f'Convergence @ epoch {conv_epoch}')
        plt.legend()
    plt.savefig(f"q3_misclassifications_Dataset_{dataset_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # decision boundary
    plt.figure(figsize=(8, 6))
    colors = {-1: '#457B9D', 1: '#E63946'}
    markers = {-1: 'o', 1: 's'}
    for label in [-1, 1]:
        mask_tr = y_train == label
        mask_te = y_test  == label
        plt.scatter(X_train[mask_tr, 0], X_train[mask_tr, 1],
                      c=colors[label], marker=markers[label],
                      alpha=0.5, s=20, label=f'Train cls {label}')
        plt.scatter(X_test[mask_te, 0], X_test[mask_te, 1],
                      c=colors[label], marker=markers[label],
                      alpha=0.9, s=40, edgecolors='k', linewidths=0.5,
                      label=f'Validation cls {label}')

    # draw boundary line:  w[0]*x1 + w[1]*x2 + b = 0
    all_x = np.vstack([X_train, X_test])
    x1_min, x1_max = all_x[:, 0].min() - 1, all_x[:, 0].max() + 1
    if abs(w[1]) > 1e-10:
        x1_vals = np.linspace(x1_min, x1_max, 300)
        x2_vals = -(w[0] * x1_vals + b) / w[1]
        plt.plot(x1_vals, x2_vals, 'k-', linewidth=2, label='Decision boundary')
    else:
        xv = -b / w[0] if abs(w[0]) > 1e-10 else 0
        plt.axvline(xv, color='k', linewidth=2, label='Decision boundary')

    test_acc = accuracy(X_test, y_test, w, b)
    title = (f"Dataset {dataset_name} – Decision Boundary\n"
             f"Test Accuracy: {test_acc:.1f}%")
    if conv_epoch:
        title += f"  |  Converged: epoch {conv_epoch}"
    else:
        title += "  |  Not converged"
    plt.title(title, fontsize=12)
    plt.xlabel("$x_1$", fontsize=11)
    plt.ylabel("$x_2$", fontsize=11)
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"q3_decision_boundary_Dataset_{dataset_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

print("=== Dataset A ===")
wA, bA, misA, convA = rosenblatt_perceptron(X_trA, y_trA, lr=0.01, max_epochs=300)
accA = accuracy(X_teA, y_teA, wA, bA)
print(f"  Test Accuracy : {accA:.2f}%")
print(f"  Final weights : w={wA}, b={bA:.4f}\n")

plot_results(X_trA, y_trA, X_teA, y_teA,
             wA, bA, misA, convA,
             dataset_name='A')

print("=== Dataset B ===")
wB, bB, misB, convB = rosenblatt_perceptron(X_trB, y_trB, lr=0.01, max_epochs=300)
accB = accuracy(X_teB, y_teB, wB, bB)
print(f"  Test Accuracy : {accB:.2f}%")
print(f"  Final weights : w={wB}, b={bB:.4f}\n")

plot_results(X_trB, y_trB, X_teB, y_teB,
             wB, bB, misB, convB,
             dataset_name='B')
print("\n========== SUMMARY ==========")
print(f"Dataset A | Convergence epoch: {convA} | Test Accuracy: {accA:.2f}%")
print(f"Dataset B | Convergence epoch: {convB} | Test Accuracy: {accB:.2f}%")