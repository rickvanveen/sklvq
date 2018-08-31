import numpy as np
import scipy as sp


def sqeuclidean(data, prototypes, *args, **kwargs):
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


def sqeuclidean_grad(data, prototype, *args, **kwargs):
    """ Implements the derivative of the squared euclidean distance.

    Parameters
    ----------
    data       : ndarray, shape(n_observations, n_features)

    prototype  : ndarray, shape(n_features,)

    Returns
    -------
    gradient : ndarray, shape(n_observations, n_features)
                The gradient with respect to the prototype and every observation in data.
    """
    return -2 * (data - prototype)


def euclidean(data, prototypes, *args, **kwargs):
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


def euclidean_grad(data, prototype, *args, **kwargs):
    """ Implements the derivative of the euclidean distance.

    Parameters
    ----------
    data       : ndarray, shape(n_observations, n_features)

    prototype  : ndarray, shape(n_features,)

    Returns
    -------
    gradient : ndarray, shape(n_observations, n_features)
               The gradient with respect to the prototype and every observation in data.
    """
    difference = data - prototype
    return (-1 * difference) / np.sqrt(np.sum(difference**2))


# TODO: Note for when working on GMLVQ -> cdist(x, y, 'mahanalobis', Omega)**2 ->
# TODO: Omega=None ...
def sqmeuclidean(data, prototypes, *args, omega=None, **kwargs):
    """ Implements a weighted variant of the squared euclidean distance.

        Note uses scipy.spatial.distance.cdist see scipy documentation.

        Note that any custom function should still accept and return the same as this function.

        Parameters
        ----------
        data       : ndarray, shape(n_obervations, n_features)
                     Inputs are converted to float type.
        prototypes : ndarray, shape(n_prototypes, n_features)
                     Inputs are converted to float type.
        omega      : ndarray, shape(n_features, n_features
                    TODO: Does not necessarily has to be square.

        Returns
        -------
        distances : ndarray, shape(n_observations, n_prototypes)
            The dist(u=XA[i], v=XB[j]) is computed and stored in the
            ij-th entry.
        """
    return sp.spatial.distance.cdist(data, prototypes, 'mahanalobis', omega)**2


def _find_min(indices, distances):
    """ Helper function to find the minimum distance and the index of this distance. """
    dist_temp = np.where(indices, distances, np.inf)
    return dist_temp.min(axis=1), dist_temp.argmin(axis=1)


def compute_distance(prototypes, p_labels, data, d_labels, distfun, distfun_kwargs):
    """ Computes the distances between each prototype and each observation and finds all indices where the smallest
    distance is that of the prototype with the same label and with a different label. """

    distances = distfun(data, prototypes, **distfun_kwargs)

    ii_same = np.transpose(np.array([d_labels == prototype_label for prototype_label in p_labels]))
    ii_diff = ~ii_same

    dist_same, i_dist_same = _find_min(ii_same, distances)
    dist_diff, i_dist_diff = _find_min(ii_diff, distances)

    return dist_same, dist_diff, i_dist_same, i_dist_diff
