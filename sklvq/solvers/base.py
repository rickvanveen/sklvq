from abc import ABC, abstractmethod
import numpy as np
from sklvq.objectives import ObjectiveBaseClass
import scipy as sp
from typing import Dict

from itertools import repeat
from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass


# TODO: Optimize, Get rid of to_variables and to_params in custom solvers.
#  Probably, this can be done by wrapping some functionallity around scipy solvers that unifies
#  the way the solvers are called, but removes the variables from the call to the objective.
class SolverBaseClass(ABC):
    def __init__(self, objective: ObjectiveBaseClass):
        self.objective = objective

    @abstractmethod
    def solve(
        self, data: np.ndarray, labels: np.ndarray, model: "LVQBaseClass",
    ) -> "LVQBaseClass":
        raise NotImplementedError("You should implement this!")

    @staticmethod
    def create_state(state_keys, **kwargs) -> dict:
        state = dict.fromkeys(state_keys)
        state.update(**kwargs)
        return state

    @staticmethod
    def multiply_model_params(
        step_sizes: Union[int, float, np.ndarray],
        model_params: Union[tuple, np.ndarray],
    ) -> Union[tuple, np.ndarray]:

        if isinstance(model_params, np.ndarray):
            return step_sizes * model_params

        if isinstance(model_params, tuple):
            if isinstance(step_sizes, int) | isinstance(step_sizes, float):
                step_sizes = repeat(step_sizes, len(model_params))
                # else isinstance(step_sizes, np.ndarray):
            return tuple(
                [
                    step_size * model_param
                    for step_size, model_param in zip(step_sizes, model_params)
                ]
            )


class ScipyBaseSolver(SolverBaseClass):
    def __init__(
        self, objective, method: str = "L-BFGS-B", params: Dict = None, **kwargs
    ):
        self.method = method
        self.params = params
        super().__init__(objective)

    def solve(
        self, data: np.ndarray, labels: np.ndarray, model: "LVQBaseClass",
    ) -> "LVQBaseClass":
        params = {"jac": self.objective.gradient}
        if self.params is not None:
            params.update(self.params)

        result = sp.optimize.minimize(
            self.objective,
            model.to_variables(model.get_model_params()),
            method=self.method,
            args=(model, data, labels),
            **params
        )

        # Update model
        model.set_model_params(model.to_params(result.x))
        return model
