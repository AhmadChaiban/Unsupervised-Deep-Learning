import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import getKaggleMNIST

def main():
    X_train, y_train, X_test, y_test = getKaggleMNIST()

    pca = PCA()
    reduced_data = pca.fit_transform(X_train)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=100, c= y_train, alpha = 0.5)
    plt.show()

    plt.plot(pca.explained_variance_ratio_)
    plt.show()

    ## calculating the cumulative variance
    ## chosse k = number of dimensions that gives us the 95 - 99% variance
    cumulative = []
    last = 0
    for v in pca.explained_variance_ratio_:
        cumulative.append(last + v)
        last = cumulative[-1]

    ## plotting the cumulative variance
    plt.plot(cumulative)
    plt.show()

if __name__ == '__main__':
    main()