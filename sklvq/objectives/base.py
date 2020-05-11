from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class ObjectiveBaseClass(ABC):
    @abstractmethod
    def __call__(
        self,
        variables: np.ndarray,
        model: "LVQClassifier",
        data: np.ndarray,
        labels: np.ndarray,
    ):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(
        self,
        variables: np.ndarray,
        model: "LVQClassifier",
        data: np.ndarray,
        labels: np.ndarray,
    ):
        raise NotImplementedError("You should implement this!")
