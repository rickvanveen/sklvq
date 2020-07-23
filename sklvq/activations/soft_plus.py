from . import ActivationBaseClass

import numpy as np


class SoftPlus(ActivationBaseClass):
    """ Soft+ function

    Function and derivatives as discussed in [1]_

    Parameters
    ----------
    beta : int or float, optional, default=1
           Parameter that can be set during instantiation in order to control the steepness
           of the constructed callable instance.

    See also
    --------
    Identity, Sigmoid, Swish

    References
    ----------
    .. [1] Villmann, T., Ravichandran, J., Villmann, A., Nebel, D., & Kaden, M. (2019). "Activation
        Functions for Generalized Learning Vector Quantization - A Performance Comparison", 2019.
    """

    def __init__(self, beta: int = 1):
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Implements the soft+ function:
            .. math::

                f(x) = \\ln(1 + e^{\\beta \\cdot x})

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of the soft+ function.
        """
        return np.log(1 + np.exp(self.beta * x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """ Implements the sigmoid function's derivative:
            .. math::

                \\frac{\\partial f}{\\partial x} = \\frac{\\beta \\cdot e^{\\beta \\cdot x}}{1 + e^{\\beta \\cdot x}}

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of the soft+ function's gradient.
        """
        exp = np.exp(self.beta * x)
        return (self.beta * exp) / (1 + exp)
