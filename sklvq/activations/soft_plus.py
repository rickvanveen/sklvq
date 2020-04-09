from . import ActivationBaseClass

import numpy as np


class SoftPlus(ActivationBaseClass):
    """ Soft+ function

    Parameters
    ----------
    beta : int, float, default=1
           Parameter that can be set during instantiation in order to control the steepness
           of the constructed callable instance.

    See also
    --------
    Identity, Sigmoid, Swish
    """

    def __init__(self, beta: int = 1):
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Implements the soft+ function:
            .. math::

                f(x) = \\ln(1 + e^{\\beta \\cdot x})

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return np.log(1 + np.exp(self.beta * x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """ Implements the sigmoid function's derivative:
            .. math::

                \\frac{\\partial f}{\\partial x} = \\frac{\\beta \\cdot e^{\\beta \\cdot x}}{1 + e^{\\beta \\cdot x}}

        Parameters
        ----------
        x : numpy.ndarray
            The values that need to be transformed.

        Returns
        -------
        numpy.ndarray
            The elementwise transformed input values
        """
        exp = np.exp(self.beta * x)
        return (self.beta * exp) / (1 + exp)
