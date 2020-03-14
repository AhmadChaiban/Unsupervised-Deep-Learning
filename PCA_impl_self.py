import numpy as np
import matplotlib.pyplot as plt
from util import getKaggleMNIST

## Let's grab all the data
X_train, y_train, X_test, y_test = getKaggleMNIST()

## calculating the covariance, then the eigenvalue decomposition
lambdas, Q = np.linalg.eigh(np.cov(X_train.T))

## Sort the lambdas in ascending order
## Note that some of them might be slighly negative due to precision,
## so we're going to remove them
idx = np.argsort(-lambdas)
## Sorting the lambdas in the proper order
lambdas = lambdas[idx]
## Getting rid of the negatives
lambdas = np.maximum(lambdas, 0)
Q = Q[:, idx]

## plotting the first two columns of Z
## Applying the basic formula here
Z = X_train.dot(Q)
plt.scatter(Z[:,0], Z[:,1], s = 100, c = y_train, alpha = 0.3)
plt.title("Plotting the first two columns of Z")
plt.show()

## plot variances
plt.plot(lambdas)
plt.title("Variance of Each component")
plt.show()

## Cumulative Variance
plt.plot(np.cumsum(lambdas))
plt.title("Cumulative Variance")
plt.show()
