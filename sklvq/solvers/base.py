from abc import ABC, abstractmethod
import numpy as np
from sklvq.objectives import ObjectiveBaseClass
import scipy as sp
from typing import Dict

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


class ScipyBaseSolver(SolverBaseClass):

    def __init__(self, method: str = 'L-BFGS-B', params: Dict = None):
        self.method = method
        self.params = params

    def solve(self, data: np.ndarray,
              labels: np.ndarray,
              objective: ObjectiveBaseClass,
              model: 'LVQClassifier') -> 'LVQClassifier':
        result = sp.optimize.minimize(
            objective,
            model.get_variables(),
            method=self.method,
            jac=objective.gradient,
            args=(model, data, labels))

        model.set_variables(result.x)
        return model