from abc import ABC, abstractmethod
import numpy as np


class ActivationBaseClass(ABC):

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("You should implement this!")
