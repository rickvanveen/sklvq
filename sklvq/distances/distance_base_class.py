from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class DistanceBaseClass(ABC):

    @abstractmethod
    def __call__(self, data: np.ndarray, model: 'LVQClassifier') -> np.ndarray:
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, data: np.ndarray, model: 'LVQClassifier', i_prototype: int) -> np.ndarray:
        raise NotImplementedError("You should implement this!")