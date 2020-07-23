from abc import ABC, abstractmethod
import numpy as np


class ActivationBaseClass(ABC):
    """ ActivationBaseClass

    Abstract class for implementing activation functions. It provides abstract methods with
    expected call signatures.

    When developing a custom activation function '__init__' should accept any parameters as
    key-value pairs.
    """

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
