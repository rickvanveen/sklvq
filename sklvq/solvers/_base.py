from abc import ABC, abstractmethod
from typing import List, Any
from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize as spo

from ..objectives import ObjectiveBaseClass

if TYPE_CHECKING:
    from ..models import LVQBaseClass


class SolverBaseClass(ABC):
    """Solver base class

    Abstract class for implementing solvers. Provides abstract methods with expected calls
    signatures.

    See also
    --------
    SteepestGradientDescent, WaypointGradientDescent, AdaptiveMomentEstimation,
    BroydenFletcherGoldfarbShanno, LimitedMemoryBfgs
    """

    def __init__(self, objective: ObjectiveBaseClass):
        self.objective = objective

    @abstractmethod
    def solve(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        model: "LVQBaseClass",
    ) -> None:
        """
        Solve updates the model it is given and does not return anything.

        Parameters
        ----------
        data : ndarray of shape (number of observations, number of dimensions)
            The data.
        labels : ndarray of size (number of observations)
            The labels of the samples in the data.
        model : LVQBaseClass
            The initial model that will also hold the final result
        """
        raise NotImplementedError("You should implement this!")


class ScipyBaseSolver(SolverBaseClass):
    """ScipyBaseSolver

    Class to wrap around scipy solvers.

    See also
    --------
    BroydenFletcherGoldfarbShanno, LimitedMemoryBfgs
    """

    def __init__(self, objective, method: str = "L-BFGS-B", **kwargs):
        self.method = method
        self.params = kwargs
        super().__init__(objective)

    def _objective_wrapper(self, variables, model, data, labels):
        model.set_variables(variables)
        return self.objective(model, data, labels)

    def _objective_gradient_wrapper(self, variables, model, data, labels):
        model.set_variables(variables)
        return self.objective.gradient(model, data, labels)

    def solve(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        model: "LVQBaseClass",
    ):
        """
        Solve updates the model it is given and does not return anything.

        Parameters
        ----------
        data : ndarray of shape (number of observations, number of dimensions)
            The data.
        labels : ndarray of size (number of observations)
            The labels of the samples in the data.
        model : LVQBaseClass
            The initial model that will also hold the final result
        """
        params = {"jac": self._objective_gradient_wrapper}
        if self.params is not None:
            params.update(self.params)

        result = spo.minimize(
            self._objective_wrapper,
            model.get_variables(),
            method=self.method,
            args=(model, data, labels),
            **params
        )

        # Update model
        model.set_variables(result.x)


def _update_state(state_keys: List[str], **kwargs: Any) -> dict:
    # Helper function that can be used to update state dict. The state_keys is a  list of strings
    # indicating the keys the dictionary should hold. If not provided in the kwargs they are set
    # to None.
    state = dict.fromkeys(state_keys)
    state.update(**kwargs)
    return state
