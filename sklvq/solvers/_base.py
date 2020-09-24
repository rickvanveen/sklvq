from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize as spo

from ..objectives import ObjectiveBaseClass

from typing import Union, List, Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import LVQBaseClass


class SolverBaseClass(ABC):
    """ SolverBaseClass

    """
    def __init__(self, objective: ObjectiveBaseClass):
        self.objective = objective

    @abstractmethod
    def solve(
        self, data: np.ndarray, labels: np.ndarray, model: "LVQBaseClass",
    ):
        """
        Solve alters the model and does not return anything

        Parameters
        ----------
        data : ndarray of shape (number of observations, number of dimensions)
        labels : ndarray of size (number of observations)
        model : LVQBaseClass
            The initial model that will hold the result
        """
        raise NotImplementedError("You should implement this!")


class ScipyBaseSolver(SolverBaseClass):

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
        self, data: np.ndarray, labels: np.ndarray, model: "LVQBaseClass",
    ):
        """
        Solve alters the model and does not return anything

        Parameters
        ----------
        data : ndarray of shape (number of observations, number of dimensions)
        labels : ndarray of size (number of observations)
        model : LVQBaseClass
            The initial model that will hold the result

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
    state = dict.fromkeys(state_keys)
    state.update(**kwargs)
    return state