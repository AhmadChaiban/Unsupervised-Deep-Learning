## Visualizing Gaussian Clouds

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

def gaussianCreator(centers, points_per_cloud):
    data = []
    for center in centers:
        cloud = np.random.randn(points_per_cloud, 3) + center
        data.append(cloud)
    return np.concatenate(data)

def plotClouds3D(points_per_cloud):
    colors = np.array([[i]*points_per_cloud for i in range(len(centers))]).flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = colors)
    plt.show()
    return colors

if __name__ == '__main__':
    ## Start by defining the centers of each Gaussian cloud
    centers = np.array([
        [ 1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [-1, -1, -1],
    ])*3

    ## Let's create the clouds, Gaussian samples centered at
    ## each of the centers we made above.
    points_per_cloud = 100
    data = gaussianCreator(centers, points_per_cloud)
    ## Visualize the clouds in 3D
    ## add colors/labels so we can track where the points go
    colors = plotClouds3D(points_per_cloud)

    ## perform dimensionality reduction with TSNE
    tsne = TSNE()
    transformed_data = tsne.fit_transform(data)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c = colors)
    plt.show()