from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..models._base import LVQBaseClass


class ObjectiveBaseClass(ABC):
    """Objective base class

    Abstract class for implementing objective functions. It provides abstract methods with
    expected call signatures.
    """

    __slots__ = ()

    @abstractmethod
    def __call__(
        self,
        model: "LVQBaseClass",
        data: np.ndarray,
        labels: np.ndarray,
    ):
        """The objective function

        Parameters
        ----------
        variables: ndarray with shape depending on model parameters
            Flattened 1D array of the variables that are changed, i.e., the model parameters

        model : LVQBaseClass
            The model which can be any LVQBaseClass compatible with this objective function.

        data: ndarray with shape (n_samples, n_features)
            The X

        labels: ndarray with shape (n_samples)


        Returns
        -------
        float
            The cost


        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(
        self,
        model: "LVQBaseClass",
        data: np.ndarray,
        labels: np.ndarray,
    ):
        """The objective gradient

        Parameters
        ----------
        variables: ndarray with shape depending on model parameters
            Flattened 1D array of the variables that are changed, i.e., the model parameters

        model : LVQBaseClass
            The model which can be any LVQBaseClass compatible with this objective function.

        data: ndarray with shape (n_samples, n_features)
            The X

        labels: ndarray with shape (n_samples)

        Returns
        -------
        ndarray with shape of variables
            The gradient

        """
        raise NotImplementedError("You should implement this!")
