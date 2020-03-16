from . import ActivationBaseClass
import numpy as np


class Swish(ActivationBaseClass):

    def __init__(self, beta: int = 1):
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Implements the swish function :cite:`Villmann2019`:
            .. math::

                f(x) = \\frac{x}{1 + e^{-\\beta \\cdot x}}

        Parameters
        ----------
        x    : ndarray
               The values that need to be transformed.

        Returns
        -------
        ndarray
            The elementwise transformed input values.
        """
        return _swish(x, self.beta)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """ Implements the sigmoid function's derivative :cite:`Villmann2019`:
            .. math::

                \\frac{\\partial f}{\\partial x} = \\beta \\cdot f(x) + (\\frac{1}{1 + e^{-\\beta \\cdot x}}) \\cdot (1 - \\beta \\cdot f(x))

        Parameters
        ----------
        x    : ndarray
               The values that need to be transformed.

        Returns
        -------
        ndarray
            The elementwise transformed input values
        """
        return (self.beta * _swish(x, self.beta)) + (_sgd(x, self.beta) * (1 - self.beta * _swish(x, self.beta)))


def _sgd(x: np.ndarray, beta: int) -> np.ndarray:
    return 1 / (np.exp(-beta * x) + 1)


def _swish(x: np.ndarray, beta: int) -> np.ndarray:
    return x * _sgd(x, beta)