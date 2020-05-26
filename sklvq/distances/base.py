from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass


class DistanceBaseClass(ABC):
    @abstractmethod
    def __call__(self, data: np.ndarray, model: "LVQBaseClass") -> np.ndarray:
        """The distance method.

        Parameters
        ----------
        data : ndarray
            The data
        model : LVQBaseClass
            The model

        Returns
        -------
        ndarray
            The distances

        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(
        self, data: np.ndarray, model: "LVQBaseClass", i_prototype: int
    ) -> np.ndarray:
        """ The distance gradient method.

        Parameters
        ----------
        data : ndarray
            The data
        model : LVQBaseClass
            The model
        i_prototype : int
            The index

        Returns
        -------
            The gradient

        """
        raise NotImplementedError("You should implement this!")
