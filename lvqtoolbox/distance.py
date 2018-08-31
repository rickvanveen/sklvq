import numpy as np
import scipy as sp


def sqeuclidean(data, prototypes):
    """ Wrapper function for scipy's cdist(x, y, 'sqeuclidean') function

    See scipy.spatial.distance.cdist for full documentation.

    Note that any custom function should still accept and return the same.

    Parameters
    ----------
    data       : ndarray, shape(n_obervations, n_features)
                 Inputs are converted to float type.
    prototypes : ndarray, shape(n_prototypes, n_features)
                 Inputs are converted to float type.

    Returns
    -------
    distances : ndarray, shape(n_observations, n_prototypes)
        The dist(u=XA[i], v=XB[j]) is computed and stored in the
        ij-th entry.
    """
    return sp.spatial.distance.cdist(data, prototypes, 'sqeuclidean')


def sqeuclidean_grad(data, prototype):
    """ Implements the derivative of the squared euclidean distance.

    Parameters
    ----------
    data       : ndarray, shape(n_observations, n_features)

    prototype  : ndarray, shape(1, n_features)

    Returns
    -------
    gradient : ndarray, shape(n_observations, n_features)
                The gradient with respect to the prototype and every observation in data.
    """
    return -2 * (data - prototype)


def euclidean(data, prototypes):
    """ Wrapper function for scipy's cdist(x, y, 'euclidean') function

    See scipy.spatial.distance.cdist for full documentation.

    Note that any custom function should still accept and return the same.

    Parameters
    ----------
    data       : ndarray, shape(n_obervations, n_features)
                 Inputs are converted to float type.
    prototypes : ndarray, shape(n_prototypes, n_features)
                 Inputs are converted to float type.

    Returns
    -------
    distances : ndarray, shape(n_observations, n_prototypes)
        The dist(u=XA[i], v=XB[j]) is computed and stored in the
        ij-th entry.
    """
    return sp.spatial.distance.cdist(data, prototypes, 'euclidean')


def euclidean_grad(data, prototype):
    """ Implements the derivative of the euclidean distance.

    Parameters
    ----------
    data       : ndarray, shape(n_observations, n_features)

    prototype  : ndarray, shape(1, n_features)

    Returns
    -------
    gradient : ndarray, shape(n_observations, n_features)
               The gradient with respect to the prototype and every observation in data.
    """
    difference = data - prototype
    return (-1 * difference) / np.sqrt(np.sum(difference**2))


# TODO: Note for when working on GMLVQ -> cdist(x, y, 'mahanalobis', Omega)**2 ->