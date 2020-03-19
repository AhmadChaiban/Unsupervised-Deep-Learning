import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from util import getKaggleMNIST

import os
import sys
sys.path.append(os.path.abspath('..'))
## Need to replace the kmeans before continuing
# from LazyProgrammer_unsupervised_class.kmeans_mnist import purity
from sklearn.mixture import GaussianMixture

def main():
    X_train, y_train, _, _ = getKaggleMNIST()
    sample_size = 1000
    X = X_train[:sample_size]
    Y = y_train[:sample_size]

    tsne = TSNE()
    Z = tsne.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1], s=100, c=Y, alpha =0.5)
    plt.show()

    ## purity measure
    ## maximum purity is 1, higher is better
    gmm = GaussianMixture(n_components=10)
    gmm.fit(X)
    Rfull = gmm.predict_proba(X)
    print("Rfull.shape:", Rfull.shape)
    # print("full purity:", purity(Y, Rfull))

    ## now try the same thing on the reduced data
    gmm.fit(Z)
    Rreduced = gmm.predict_proba(Z)
    print("reduced purity:", purity(Y, Rreduced))

if __name__ == '__main__':
    main()
