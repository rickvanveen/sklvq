from abc import ABC, abstractmethod

import numpy as np

ABC_METHOD_NOT_IMPL_MSG = "You should implement this!"

class DiscriminantBaseClass(ABC):
    """Discriminant base class

    Abstract class for implementing discriminant functions. Provides abstract methods with
    expected call signatures.

    Custom discriminative function '__init__' should accept any parameters as key-value pairs.

    See also
    --------
    RelativeDistance
    """

    __slots__ = ()

    @abstractmethod
    def __call__(self, dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
        """
        Should implement a discriminant function

        Parameters
        ----------
        dist_same : ndarray of shape (n_samples, 1), with n_samples >= 1
            Shortest distance of n_samples to a prototype with the same label.
        dist_diff : ndarray of shape (n_samples, 1), with n_samples >= 1
            Shortest distance of n_samples to a prototype with a different label.

        Returns
        -------
        ndarray : with shape (n_samples, 1)
            Should perform a elementwise evaluation of a discriminant function.
        """
        raise NotImplementedError(ABC_METHOD_NOT_IMPL_MSG)

    @abstractmethod
    def gradient(
        self, dist_same: np.ndarray, dist_diff: np.ndarray, *, same_label: bool
    ) -> np.ndarray:
        """
        Should implement the discriminant function's  gradient

        Parameters
        ----------
        dist_same : ndarray with shape (n_samples, 1), with n_samples >= 1
            Shortest distance of n_samples to a prototype with the same label.
        dist_diff : ndarray with shape (n_samples, 1), with n_samples >= 1
            Shortest distance of n_samples to a prototype with a different label.
        same_label : bool
            Indicates if the gradient with respect to a prototype with the same label (True) or
            with respect to a prototype with a different label (False) needs to be computed.

        Returns
        -------
        ndarray with shape (n_sampeles, 1)
            Should perform a elementwise evaluation of a discriminant function's gradient.

        """
        raise NotImplementedError(ABC_METHOD_NOT_IMPL_MSG)
