import numpy as np


def sigmoid(x, beta=2):
    """ Implements the sigmoid function: f(x) = 1 /( e^{-Bx} + 1)

    Parameters
    ----------
    x    : ndarray
           The values that need to be scaled.
    beta : scalar
           Controls the slope of the sigmoid function.

    Returns
    -------
    scaled_x : ndarray, same shape as input
               The elementwise scaled input values.
    """
    return 1 / (np.exp(-beta * x) + 1)


def sigmoid_grad(x, beta=2):
    return (beta * np.exp(beta*x)) / (np.exp(beta*x) + 1)**2


def identity(x, **kwargs):
    """ Implements the sigmoid function: f(x) = 1 /( e^{-Bx} + 1)

    Parameters
    ----------
    x    : ndarray
           The values that need to be scaled.
    beta : scalar
           Controls the slope of the sigmoid function.

    Returns
    -------
    scaled_x : ndarray, same shape as input
               The elementwise scaled input values.
    """
    return x


def identity_grad(*args, **kwargs):
    return 1
