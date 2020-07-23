from abc import ABC, abstractmethod
import numpy as np


class DiscriminativeBaseClass(ABC):
    """ DiscriminativeBaseClass

    Abstract class for implementing discriminant functions. It provides abstract methods with
    expected call signatures.

    When developing a custom activation function '__init__' should accept any parameters as
    key-value pairs.
    """

    @abstractmethod
    def __call__(self, dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        dist_same : ndarray with shape (n_samples, 1)
            Shortest distance of one or more samples to a prototype with the same label.
        dist_diff : ndarray with shape (n_samples, 1)
            Shortest distance of one or more samples to a prototype with a different label.

        Returns
        -------
            ndarray : with shape (n_samples, 1)
                Elementwise evaluation of the discriminative function
        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(
        self, dist_same: np.ndarray, dist_diff: np.ndarray, same_label: bool
    ) -> np.ndarray:
        """

        Parameters
        ----------
        dist_same : ndarray with shape (n_samples, 1)
            Shortest distance of one or more samples to a prototype with the same label.
        dist_diff : ndarray with shape (n_samples, 1)
            Shortest distance of one or more samples to a prototype with a different label.
        same_label : bool
            Indicates if the gradient with respect to a prototype with the same label (True) or
            with respect to a prototype with a different label (False) needs to be computed.

        Returns
        -------
            ndarray with shape (n_sampeles, 1)
                Elementwise evaluation of the discriminative function's gradient.

        """
        raise NotImplementedError("You should implement this!")
