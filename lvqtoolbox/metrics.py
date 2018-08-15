import numpy as np


def squared_euclidean(x, y):
    return np.sum((x - y) ** 2)


def squared_euclidean_grad(x, y):
    return -2 * (x - y)


def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def euclidean_grad(x, y):
    difference = x - y
    return (-1 * difference) / np.sqrt(np.sum(difference**2))