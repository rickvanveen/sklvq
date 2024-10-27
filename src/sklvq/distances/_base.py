from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..models import LVQBaseClass


class DistanceBaseClass(ABC):
    """Distance base class

    Abstract class for implementing distance functions. It provides abstract methods with
    expected call signatures.

    Custom distance function '__init__' should accept any parameters as key-value pairs.

    See also
    --------
    Euclidean, SquaredEuclidean, AdaptiveSquaredEuclidean, LocalAdaptiveSquaredEuclidean
    """

    __slots__ = ()

    @abstractmethod
    def __call__(self, data: np.ndarray, model: "LVQBaseClass") -> np.ndarray:
        """
        Should implement a distance function.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The samples for which the distance to the prototypes of the model need to be computed.
        model : LVQBaseClass
            Any class extending the LVQBaseClass or depending on the type of distance function
            is implemented a class that provides the required attributes.

        Returns
        -------
        ndarray with shape (n_samples, n_prototypes)
            Evaluation of the distance between each sample and prototype of the model.
        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(
        self, data: np.ndarray, model: "LVQBaseClass", i_prototype: int
    ) -> np.ndarray:
        """The distance gradient method.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : LVQBaseClass
            Any class extending the LVQBaseClass or depending on the type of distance function
            is implemented, a class that provides the required attributes.
        i_prototype : int
            The index of the prototype for which the gradient needs to be computed.

        Returns
        -------
        ndarray with shape (n_samples, n_features)
            The gradient with respect to the prototype (i_prototype) and every sample in X.

        """
        raise NotImplementedError("You should implement this!")
