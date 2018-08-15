import numpy as np


# Use cdist build in is much much faster
def squared_euclidean(x, y):
    return np.sum((x - y) ** 2)


def squared_euclidean_grad(x, y):
    return -2 * (x - y)

# Use cdist build in is much much faster
def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def euclidean_grad(x, y):
    difference = x - y
    return (-1 * difference) / np.sqrt(np.sum(difference**2))

# TODO: Note for when working on GMLVQ -> cdist(x, y, 'mahanalobis', Omega)**2 ->