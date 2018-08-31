import numpy as np
import scipy as sp


def sqeuclidean(x, y):
    """ Wrapper function for scipy's cdist(x, y, 'sqeuclidean') function

    See scipy.spatial.distance.cdist for full documentation.

    Note that any custom function should still accept and return the same.

    Parameters
    ----------
    x : ndarray
        An :math:`m_x` by :math:`n` array of :math:`m_x`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    y : ndarray
        An :math:`m_y` by :math:`n` array of :math:`m_y`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.

    Returns
    -------
    d : ndarray
        A :math:`m_x` by :math:`m_y` distance matrix is returned.
        For each :math:`i` and :math:`j`, the metric
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.

    """
    return sp.spatial.distance.cdist(x, y, 'sqeuclidean')


def sqeuclidean_grad(x, y):
    """ Implements the derivative of the squared euclidean distance.

    Parameters
    ----------
    x : ndarray
        An :math: 'm_x' by :math: 'n' array of :math: 'm_x"
        original observations in an :math:`n`-dimensional space.
    y : ndarray
        A single observation by 'n' array in :math: 'n'-dimensional space.

    Returns
    -------
    g : ndarray
        The :math: 'm_x' by :math: 'n' array of :math: 'm_x'
        by :math: 'n' partial derivatives values.
    """
    return -2 * (x - y)


def euclidean(x, y):
    """ Wrapper function for scipy's cdist(x, y, 'euclidean') function

    See scipy.spatial.distance.cdist for full documentation.

    Note that any custom function should still accept and return the same.

    Parameters
    ----------
    x : ndarray
        An :math:`m_x` by :math:`n` array of :math:`m_x`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    y : ndarray
        An :math:`m_y` by :math:`n` array of :math:`m_y`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.

    Returns
    -------
    d : ndarray
        A :math:`m_x` by :math:`m_y` distance matrix is returned.
        For each :math:`i` and :math:`j`, the metric
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.

    """
    return sp.spatial.distance.cdist(x, y, 'euclidean')


def euclidean_grad(x, y):
    """ Implements the derivative of the euclidean distance.

    Parameters
    ----------
    x : ndarray
        An :math: 'm_x' by :math: 'n' array of :math: 'm_x"
        original observations in an :math:`n`-dimensional space.
    y : ndarray
        A single observation by 'n' array in :math: 'n'-dimensional space.

    Returns
    -------
    g : ndarray
        The :math: 'm_x' by :math: 'n' array of :math: 'm_x'
        by :math: 'n' partial derivative values.
    """
    difference = x - y
    return (-1 * difference) / np.sqrt(np.sum(difference**2))


# TODO: Note for when working on GMLVQ -> cdist(x, y, 'mahanalobis', Omega)**2 ->