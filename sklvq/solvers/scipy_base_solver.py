from . import SolverBaseClass
from typing import Dict
import scipy as sp
import numpy as np

from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


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
