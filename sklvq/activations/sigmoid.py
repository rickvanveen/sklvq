from . import ActivationBaseClass
import numpy as np

from typing import Union


class Sigmoid(ActivationBaseClass):
    """ Sigmoid function

    Parameters
    ----------
    beta : int, float, default=1
           Parameter that can be set during instantiation in order to control the steepness
           of the constructed callable instance.

    See also
    --------
    Identity, SoftPlus, Swish
    """

    def __init__(self, beta: Union[int, float] = 1):
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Computes the sigmoid function:
            .. math::

                f(x) = \\frac{1}{e^{-\\beta \\cdot x} + 1}

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of the sigmoid function.
        """
        return np.asarray(1 / (np.exp(-self.beta * x) + 1))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """ Computes the sigmoid function's derivative with respect to x:
            .. math::

                \\frac{\\partial f}{\\partial x} = \\frac{(\\beta \\cdot e^{\\beta \\cdot x)}}{(e^{\\beta \\cdot x} + 1)^2}

        Parameters
        ----------
         x : ndarray

        Returns
        ------- 
        ndarray of shape (x.shape)
            Elementwise evaluation of the sigmoid function's gradient.
        """
        exp = np.exp(self.beta * x)
        return np.asarray((self.beta * exp) / (exp + 1) ** 2)
