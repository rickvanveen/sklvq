from abc import ABC, abstractmethod
import numpy as np


class DiscriminativeBaseClass(ABC):
    @abstractmethod
    def __call__(self, dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(
        self, dist_same: np.ndarray, dist_diff: np.ndarray, winner_same: bool
    ) -> np.ndarray:
        raise NotImplementedError("You should implement this!")
