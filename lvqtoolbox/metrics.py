import numpy as np


def squared_euclidean(v1, v2):
    return np.sum((v1 - v2) ** 2)


def squared_euclidean_gradient(v1, v2):
    return -2 * (v2 - v1)


def euclidean(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


def euclidean_gradient(v1, v2):
    difference = v2 - v1
    return (-1 * difference) / np.sqrt(np.sum(difference ** 2))