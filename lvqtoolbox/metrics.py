import numpy as np


def squared_euclidean(v1, v2):
    return np.sum((v1 - v2) ** 2)


def euclidean(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))
