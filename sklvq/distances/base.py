from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class DistanceBaseClass(ABC):
    @abstractmethod
    def __call__(self, data: np.ndarray, model: "LVQClassifier") -> np.ndarray:
        """The distance method.

        Parameters
        ----------
        data : ndarray
            The data
        model : LVQClassifier
            The model

        Returns
        -------
        ndarray
            The distances

        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(
        self, data: np.ndarray, model: "LVQClassifier", i_prototype: int
    ) -> np.ndarray:
        """ The distance gradient method.

        Parameters
        ----------
        data : ndarray
            The data
        model : LVQClassifier
            The model
        i_prototype : int
            The index

        Returns
        -------
            The gradient

        """
        raise NotImplementedError("You should implement this!")
