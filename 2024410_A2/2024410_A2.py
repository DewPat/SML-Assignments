import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

np.random.seed(0)
# data fetching 
def load_mnist_subset():
    mnist = fetch_openml('mnist_784', version=1, parser='liac-arff', as_frame=False)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)

    Xtr_full, Xte_full = X[:60000], X[60000:]
    ytr_full, yte_full = y[:60000], y[60000:]

    classes = [0,1,2]

    def sample(X,y):
        Xo, yo = [], []
        for c in classes:
            idx = np.where(y==c)[0]
            sel = np.random.choice(idx,100,replace=False)
            Xo.append(X[sel])
            yo.append(y[sel])
        return np.vstack(Xo), np.hstack(yo)

    Xtr, ytr = sample(Xtr_full,ytr_full)
    Xte, yte = sample(Xte_full,yte_full)

    return Xtr.T, ytr, Xte.T, yte

#pca matrix from eigenvectors
def PCA(X, var):
    mu = np.mean(X,axis=1,keepdims=True)
    Xc = X - mu
    S = (Xc @ Xc.T)/(X.shape[1]-1)

    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    cum = np.cumsum(eigvals)/np.sum(eigvals)
    k = np.searchsorted(cum,var)+1

    return eigvecs[:,:k], mu

def PCA_Project(X,U,mu):
    return U.T @ (X-mu)

def PCA_Reconstruct(Y,U,mu):
    return U @ Y + mu


# matrix w is formed for fda
def FDA(X,y,reg=0.1):

    classes = np.unique(y)
    d = X.shape[0]
    mu = np.mean(X,axis=1,keepdims=True)
    SB = np.zeros((d,d))
    SW = np.zeros((d,d))

    for c in classes:
        Xc = X[:,y==c]
        muc = np.mean(Xc,axis=1,keepdims=True)
        Nc = Xc.shape[1]
        SB += Nc*(muc-mu)@(muc-mu).T
        diff = Xc - muc
        SW += diff@diff.T

    SW += reg*np.eye(d)
    mat = np.linalg.solve(SW,SB)
    eigvals, eigvecs = np.linalg.eig(mat)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:,idx[:len(classes)-1]]

def FDA_Project(X,W):
    return W.T @ X



# multivariate guassian discriminant after taking log
def MVG_Discriminant(x,mean,cov):
    sign, logdet = np.linalg.slogdet(cov)
    diff = x-mean
    sol = np.linalg.solve(cov, diff)
    return -0.5*(logdet + diff.T@sol)


# mean, covaraince, prior for lda
def LDA(X,y):
    classes = np.unique(y)
    means, priors = {}, {}
    cov = np.zeros((X.shape[0],X.shape[0]))

    for c in classes:
        Xc = X[:,y==c]
        means[c] = np.mean(Xc,axis=1)
        priors[c] = Xc.shape[1]/X.shape[1]
        diff = Xc - means[c].reshape(-1,1)
        cov += diff@diff.T

    cov /= (X.shape[1]-len(classes))
    cov += 0.1*np.eye(cov.shape[0])

    covs = {c: cov for c in classes}
    return means, priors, covs


# mean, covarainces, prior for qda
def QDA(X,y):
    classes = np.unique(y)
    means, priors, covs = {}, {}, {}

    for c in classes:
        Xc = X[:,y==c]
        priors[c] = Xc.shape[1]/X.shape[1]
        means[c] = np.mean(Xc,axis=1)
        diff = Xc - means[c].reshape(-1,1)
        cov = (diff@diff.T)/(Xc.shape[1]-1)
        cov += 0.1*np.eye(cov.shape[0])
        covs[c] = cov

    return means, priors, covs


# discriminant analysis
def Discriminant(model,X):
    means, priors, covs = model
    classes = list(means.keys())
    preds = []

    for i in range(X.shape[1]):
        x = X[:,i]
        scores = [
            MVG_Discriminant(x,means[c],covs[c]) + np.log(priors[c])
            for c in classes
        ]
        preds.append(classes[np.argmax(scores)])
    return np.array(preds)

# accuracy calculation
def evaluate(model,Xtr,ytr,Xte,yte):
    tr = np.mean(Discriminant(model,Xtr)==ytr)
    te = np.mean(Discriminant(model,Xte)==yte)
    return float(tr),float(te)


# main function recall
Xtr,ytr,Xte,yte = load_mnist_subset()
Up75,mu75 = PCA(Xtr,0.75)
Ytr75 = PCA_Project(Xtr,Up75,mu75)
Yte75 = PCA_Project(Xte,Up75,mu75)

Up90,mu90 = PCA(Xtr,0.90)
Ytr90 = PCA_Project(Xtr,Up90,mu90)
Yte90 = PCA_Project(Xte,Up90,mu90)
Up2,mu2 = PCA(Xtr,0.999)
Up2 = Up2[:,:2]
Ytr2 = PCA_Project(Xtr,Up2,mu2)
Yte2 = PCA_Project(Xte,Up2,mu2)

W = FDA(Xtr,ytr)
Ztr = FDA_Project(Xtr,W)
Zte = FDA_Project(Xte,W)
Ysample = Ytr75[:,:5]
Xrec = PCA_Reconstruct(Ysample,Up75,mu75)
Xorig = Xtr[:,:5]
mse = np.mean((Xorig-Xrec)**2)

# terminal outputs
print("Raw LDA:", evaluate(LDA(Xtr,ytr), Xtr,ytr,Xte,yte))
print("Raw QDA:", evaluate(QDA(Xtr,ytr), Xtr,ytr,Xte,yte))
print("PCA 75% + LDA:", evaluate(LDA(Ytr75,ytr), Ytr75,ytr,Yte75,yte))
print("PCA 75% + QDA:", evaluate(QDA(Ytr75,ytr), Ytr75,ytr,Yte75,yte))
print("PCA 90% + LDA:", evaluate(LDA(Ytr90,ytr), Ytr90,ytr,Yte90,yte))
print("PCA 90% + QDA:", evaluate(QDA(Ytr90,ytr), Ytr90,ytr,Yte90,yte))
print("First 2 PC + LDA:", evaluate(LDA(Ytr2,ytr), Ytr2,ytr,Yte2,yte))
print("FDA + LDA:", evaluate(LDA(Ztr,ytr), Ztr,ytr,Zte,yte))
print("FDA + QDA:", evaluate(QDA(Ztr,ytr), Ztr,ytr,Zte,yte))
print("Reconstruction Mean Squared Error :", mse)

#results reconstruction
fig, ax = plt.subplots(2,5, figsize=(10,4))
for i in range(5):
    ax[0,i].imshow(Xorig[:,i].reshape(28,28), cmap='gray')
    ax[0,i].set_title("Original")
    ax[0,i].axis('off')

    ax[1,i].imshow(Xrec[:,i].reshape(28,28), cmap='gray')
    ax[1,i].set_title("Reconstructed")
    ax[1,i].axis('off')

plt.tight_layout()
plt.show()

#plots
fig, ax = plt.subplots(1,2, figsize=(10,4))
for c in [0,1,2]:
    idx = ytr==c
    ax[0].scatter(Ytr2[0,idx],Ytr2[1,idx],label=str(c))
ax[0].legend()
ax[0].set_title("PCA")
for c in [0,1,2]:
    idx = ytr==c
    ax[1].scatter(Ztr[0,idx],Ztr[1,idx],label=str(c))
ax[1].legend()
ax[1].set_title("FDA")
plt.tight_layout()
plt.show()