import numpy as np
from sklearn.decomposition import PCA
from future.utils import iteritems
# from sklearn.naive_bayes import GaussianNB
# ## Doesn't have smoothing so we're going to build it
from scipy.stats import multivariate_normal as mvn
from util import getKaggleMNIST


class GaussianNB(object):
    def fit(self, X, Y, smoothing = 1e-2):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis = 0),
                'var' : current_x.var(axis = 0) + smoothing
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, var = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X, mean = mean, cov = var) + np.log(self.priors[c])
        return np.argmax(P, axis = 1)


## Get the data
X_train, y_train, X_test, y_test = getKaggleMNIST()

## try NB by itself
model1 = GaussianNB()
model1.fit(X_train, y_train)
print("NB train score:", model1.score(X_train, y_train))
print("NB test score:", model1.score(X_test, y_test))

## try NB with PCA first
pca = PCA(n_components = 50)
Z_train = pca.fit_transform(X_train)
Z_test = pca.transform(X_test)

model2 = GaussianNB()
model2.fit(Z_train, y_train)

print(f"NB with PCA train score: {model2.score(Z_train, y_train)}")
print(f"NB with PCA test score: {model2.score(Z_test, y_test)}")

