# TODO: MEtrics is not really a good name... distance...

import numpy as np
import scipy as sp


# Use cdist build in is much much faster
def sqeuclidean(x, y):
    # return np.sum((x - y) ** 2)
    return sp.spatial.distance.cdist(x, y, 'sqeuclidean')


def sqeuclidean_grad(x, y):
    return -2 * (x - y)


# Use cdist build in is much much faster
def euclidean(x, y):
    # return np.sqrt(np.sum((x - y) ** 2))
    return sp.spatial.distance.cdist(x, y, 'euclidean')


def euclidean_grad(x, y):
    difference = x - y
    return (-1 * difference) / np.sqrt(np.sum(difference**2))

# TODO: Note for when working on GMLVQ -> cdist(x, y, 'mahanalobis', Omega)**2 ->