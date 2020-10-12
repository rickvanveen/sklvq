import numpy as np

from abc import ABC, abstractmethod


class ActivationBaseClass(ABC):
    """ ActivationBaseClass

    Abstract class for implementing activation functions, providing abstract methods with
    expected call signatures.

    See also
    --------
    Identity, Sigmoid, SoftPlus, Swish
    """
    __slots__ = ()

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : ndarray of any shape

        Returns
        -------
        ndarray of shape (x.shape)
            Should perform an elementwise evaluation of some activation function.

        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x : ndarray of any shape

        Returns
        -------
        ndarray of shape (x.shape)
            Should return the elementwise evaluation of the activation function.

        """
        raise NotImplementedError("You should implement this!")