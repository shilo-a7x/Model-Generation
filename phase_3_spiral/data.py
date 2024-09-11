# Code for creating a spiral dataset from CS231n
import numpy as np
import matplotlib.pyplot as plt


# # lets visualize the data
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
# plt.show()


def create_spiral(N=100, K=3, gamma=0.5):
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    # N = 100  # number of points per class
    D = 2  # dimensionality
    # K = 3  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype="uint8")  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * gamma  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


# X, y = create_spiral(gamma=0.1)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
# plt.show()
