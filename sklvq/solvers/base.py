from abc import ABC, abstractmethod
import numpy as np
from sklvq.objectives import ObjectiveBaseClass
import scipy as sp
from typing import Dict

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklvq.models import LVQClassifier

# TODO: vSGD, Adam, WA-GD (LeKander)


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

        params = {'jac': objective.gradient}
        if self.params is not None:
            params.update(self.params)

        result = sp.optimize.minimize(
            objective,
            model.to_variables(model.get_model_params()),
            method=self.method,
            args=(model, data, labels),
            **params)

        # Update model
        model.set_model_params(model.to_params(result.x))
        return model
