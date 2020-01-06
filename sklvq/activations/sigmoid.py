from . import ActivationBaseClass
import numpy as np


class Sigmoid(ActivationBaseClass):

    def __init__(self, beta: int = 1):
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Implements the sigmoid function :cite:`Villmann2019`:
            .. math::

                f(x) = \\frac{1}{e^{-\\beta \\cdot x} + 1}

        Parameters
        ----------
        x    : ndarray
               The values that need to be transformed.

        Returns
        -------
        ndarray
            The elementwise transformed input values.
        """
        return np.asarray(1 / (np.exp(-self.beta * x) + 1))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """ Implements the sigmoid function's derivative :cite:`Villmann2019`:
            .. math::

                g(x) = \\frac{(\\beta \\cdot e^{\\beta \\cdot x)}}{(e^{\\beta \\cdot x} + 1)^2}

        Parameters
        ----------
         x    : ndarray
               The values that need to be transformed.

        Returns
        -------
        ndarray
                The elementwise transformed input values
        """
        exp = np.exp(self.beta * x)
        return np.asarray((self.beta * exp) / (exp + 1) ** 2)
