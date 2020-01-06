from . import ActivationBaseClass

import numpy as np


class SoftPlus(ActivationBaseClass):

    def __init__(self, beta: int = 1):
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Implements the soft+ function :cite:`Villmann2019`:
            .. math::

                f(x) = \\ln(1 + e^{\\beta \\cdot x})

        Parameters
        ----------
        x    : ndarray
               The values that need to be transformed.

        Returns
        -------
        ndarray
            The elementwise transformed input values.
        """
        return np.log(1 + np.exp(self.beta * x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """ Implements the sigmoid function's derivative :cite:`Villmann2019`:
            .. math::

                g(x) = \\frac{\\beta \\cdot e^{\\beta \\cdot x}}{1 + e^{\\beta \\cdot x}}

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
        return (self.beta * exp) / (1 + exp)
