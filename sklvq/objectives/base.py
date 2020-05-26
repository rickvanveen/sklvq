from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass


class ObjectiveBaseClass(ABC):
    @abstractmethod
    def __call__(
        self,
        variables: np.ndarray,
        model: "LVQBaseClass",
        data: np.ndarray,
        labels: np.ndarray,
    ):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(
        self,
        variables: np.ndarray,
        model: "LVQBaseClass",
        data: np.ndarray,
        labels: np.ndarray,
    ):
        raise NotImplementedError("You should implement this!")
