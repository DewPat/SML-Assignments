import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# data preprocessing
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16)) 
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8)) 
        return np.fromfile(f, dtype=np.uint8)

def prepare_dataset(images,labels,classes=[0,1,2],samples_per_digit=100):
    images_arr,labels_arr = [], []
    for c in classes:
        indices = np.where(labels == c)[0]
        selected = np.random.choice(indices,samples_per_digit,replace=False)
        images_arr.append(images[selected])
        labels_arr.append(labels[selected])
    
    return np.vstack(images_arr) / 255.0, np.hstack(labels_arr)

# load raw files
train_img = load_mnist_images('train-images.idx3-ubyte')
train_lbl = load_mnist_labels('train-labels.idx1-ubyte')
test_img = load_mnist_images('t10k-images.idx3-ubyte')
test_lbl = load_mnist_labels('t10k-labels.idx1-ubyte')

X_train, y_train = prepare_dataset(train_img, train_lbl)
X_test, y_test = prepare_dataset(test_img, test_lbl)


# mean and covariance calculation
def MLE(images,labels,classes):
    means,covs,priors ={},{},{}
    for c in classes:
        X_c = images[labels == c]
        N_c = X_c.shape[0]
       
        mu_c = np.mean(X_c, axis=0)
        
        
        diff = X_c - mu_c
        sigma_c = (diff.T @ diff) / N_c
        sigma_c += np.eye(sigma_c.shape[0]) * 0.001
        
        means[c] = mu_c
        covs[c] = sigma_c
        priors[c] = N_c / len(images)
        
    return means, covs, priors

classes = [0, 1, 2]
means, covs, priors = MLE(X_train, y_train, classes)

# lda and qda implementation~
def Discriminant_Analysis(X, means, covs, priors, classes, mode='LDA'):
    pooled_cov = np.mean(list(covs.values()), axis=0)
    pooled_inv = np.linalg.inv(pooled_cov)
    predictions = []
    discriminant_values = []
    
    for x in X:
        scores = {}
        for c in classes:
            mu_c = means[c]
            pi_c = priors[c]
            
            if mode == 'LDA':
                score = x.T @ pooled_inv @ mu_c - 0.5 * mu_c.T @ pooled_inv @ mu_c + np.log(pi_c)
            else:
                sigma_inv = np.linalg.inv(covs[c])
                noshoot, logdet = np.linalg.slogdet(covs[c])
                diff = x - mu_c
                score = -0.5 * logdet - 0.5 * diff.T @ sigma_inv @ diff + np.log(pi_c)
            
            scores[c] = score
        
        preds = max(scores,key=scores.get)
        predictions.append(preds)
        discriminant_values.append(scores)
        
    return np.array(predictions), discriminant_values

# predict and calculate accuracy
lda_preds, lda_scores = Discriminant_Analysis(X_test, means, covs, priors, classes, 'LDA')
qda_preds, qda_scores = Discriminant_Analysis(X_test, means, covs, priors, classes, 'QDA')

lda_acc = np.mean(lda_preds == y_test)
qda_acc = np.mean(qda_preds == y_test)


print(f"--- Results ---")
print(f"LDA Accuracy: {lda_acc * 100:.4f}%")
print(f"QDA Accuracy: {qda_acc * 100:.4f}%")
print(f"\nSample Discriminant Values:")
print(f"True Label: {y_test[0]}")
print(f"\nLDA Discriminant Scores:")
for digit, score in lda_scores[0].items():
    marker = " ← PREDICTED" if digit == lda_preds[0] else ""
    print(f"  Digit {digit}: {float(score):12.4f}{marker}")
print(f"\nQDA Discriminant Scores:")
for digit, score in qda_scores[0].items():
    marker = " ← PREDICTED" if digit == qda_preds[0] else ""
    print(f"  Digit {digit}: {float(score):12.4f}{marker}")

#  t-SNE Visualizations
def plot_tsne_comparison(X_train, y_train, X_test, y_test):
    tsne = TSNE(n_components=2, random_state=42)
    plt.figure(figsize=(14, 6))

    # train plot
    X_train_2d = tsne.fit_transform(X_train)
    plt.subplot(1, 2, 1)
    for c in [0, 1, 2]:
        idx = np.where(y_train == c)
        plt.scatter(X_train_2d[idx, 0], X_train_2d[idx, 1], label=f'Digit {c}', alpha=0.7)
    plt.title("Train Set")
    plt.legend()
    
    # test plot
    X_test_2d = tsne.fit_transform(X_test)
    plt.subplot(1, 2, 2)
    for c in [0, 1, 2]:
        idx = np.where(y_test == c)
        plt.scatter(X_test_2d[idx, 0], X_test_2d[idx, 1], label=f'Digit {c}', alpha=0.7)
    plt.title("Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

plot_tsne_comparison(X_train, y_train, X_test, y_test)