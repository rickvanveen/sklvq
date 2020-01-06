from . import ActivationBaseClass
import numpy as np


class Identity(ActivationBaseClass):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Implements the identity function:
            .. math::

                f(x) = x

        Parameters
        ----------
        x : ndarray
            Anything really.

        Returns
        -------
        x : ndarray
            Effectively does nothing.
        """
        return x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Implements the identity function's derivative:
            .. math::

                g(x) = 1

        Parameters
        ----------
        x : ndarray
            Anything really.

        Returns
        -------
        1 : int
        """
        return np.ones(x.shape)
