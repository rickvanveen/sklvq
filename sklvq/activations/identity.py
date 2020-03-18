from . import ActivationBaseClass
import numpy as np


class Identity(ActivationBaseClass):
    """ Callable Identity function

    See also
    --------
    Sigmoid, SoftPlus, Swish
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Implements the identity function:
            .. math::

                f(x) = x

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        x : numpy.ndarray
        """
        return x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Implements the identity function's derivative:
            .. math::

                \\frac{\\partial f}{\\partial x} = 1

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Output has the same size and shape as the input and contains only 1's.
        """
        return np.ones(x.shape)
