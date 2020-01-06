from abc import ABC, abstractmethod
import numpy as np
from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class SolverBaseClass(ABC):

    @abstractmethod
    def solve(self, data: np.ndarray,
              labels: np.ndarray,
              objective: ObjectiveBaseClass,
              model: 'LVQClassifier') -> 'LVQClassifier':
        raise NotImplementedError("You should implement this!")
