import numpy as np


def sigmoid(x, beta=2):
    """ Implements the sigmoid function: f(x) = 1 /( e^{-beta * x} + 1)

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
    """ Implements the sigmoid function's derivative: g(x) = (beta * e^(beta * x)) / (e^(beta * x) + 1)^2

    Parameters
    ----------
     x    : ndarray
           The values that need to be scaled.

    Returns
    -------
    gradient : ndarray, same shape as input
               The elementwise scalded input values
    """
    return (beta * np.exp(beta*x)) / (np.exp(beta*x) + 1)**2


def identity(x, **kwargs):
    """ Implements the identity function: f(x) = x

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    x : ndarray, same shape as input
        Exactly the same as the input.
    """
    return x


def identity_grad(*args, **kwargs):
    """ Implements the identity function derivative: g(x) = 1

    Parameters
    ----------
    args, kwargs : Anything

    Returns
    -------
    gradient : scalar
               Returns the constant 1 no matter the shape or values of the input.
    """
    return 1
