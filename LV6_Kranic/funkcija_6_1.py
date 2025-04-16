import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np

def generate_data(n_samples, flagc):
    if flagc == 1:
        random_state = 365
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    elif flagc == 2:
        random_state = 148
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples,
                                    centers=4,
                                    cluster_std=[1.0, 2.5, 0.5, 3.0],
                                    random_state=random_state)
    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    else:
        X = []
    return X

n_samples = 500
flagc = 3 

X = generate_data(n_samples, flagc)

n_clusters = 4 if flagc == 3 else 3

kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centar klastera')
plt.title(f'KMeans grupiranje (flagc={flagc})')
plt.legend()
plt.grid(True)
plt.show()
