from abc import ABC, abstractmethod
import numpy as np


class ActivationBaseClass(ABC):
    """ ActivationBaseClass

    Abstract class for implementing activation functions. It provides abstract methods with
    correct call signatures.

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
            Some elementwise transformation of x.

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
            Derivative evaluated at each value in x.

        """
        raise NotImplementedError("You should implement this!")
