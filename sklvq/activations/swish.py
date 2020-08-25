from . import ActivationBaseClass
import numpy as np


class Swish(ActivationBaseClass):
    """ Swish function

    Function and derivatives as discussed in [1]_

    Parameters
    ----------
    beta : int, float, default=1
           Parameter that can be set during instantiation in order to control the steepness
           of the constructed callable instance.

    See also
    --------
    Identity, Sigmoid, SoftPlus

    References
    ----------
    .. [1] Villmann, T., Ravichandran, J., Villmann, A., Nebel, D., & Kaden, M. (2019). "Activation
        Functions for Generalized Learning Vector Quantization - A Performance Comparison", 2019.
    """

    __slots__ = "beta"

    def __init__(self, beta: int = 1):
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Implements the swish function:
            .. math::

                f(x) = \\frac{x}{1 + e^{-\\beta \\cdot x}}

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of the swish function.
        """
        return _swish(x, self.beta)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """ Implements the sigmoid function's derivative:
            .. math::

                \\frac{\\partial f}{\\partial x} = \\beta \\cdot f(x) + (\\frac{1}{1 + e^{-\\beta \\cdot x}}) \\cdot (1 - \\beta \\cdot f(x))

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of the swish function's gradient.
        """
        return (self.beta * _swish(x, self.beta)) + (
            _sgd(x, self.beta) * (1 - self.beta * _swish(x, self.beta))
        )


def _sgd(x: np.ndarray, beta: int) -> np.ndarray:
    return 1 / (np.exp(-beta * x) + 1)


def _swish(x: np.ndarray, beta: int) -> np.ndarray:
    return x * _sgd(x, beta)
