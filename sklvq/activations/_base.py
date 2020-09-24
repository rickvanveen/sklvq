import numpy as np

from abc import ABC, abstractmethod


class ActivationBaseClass(ABC):
    """ ActivationBaseClass

    Abstract class for implementing activation functions.

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
        x : ndarray

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of an activation function.

        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of an activation function's gradient.

        """
        raise NotImplementedError("You should implement this!")