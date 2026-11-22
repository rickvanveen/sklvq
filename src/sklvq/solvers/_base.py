from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.optimize as spo

if TYPE_CHECKING:
    from sklvq.models._base import LVQBaseClass
    from sklvq.objectives._base import ObjectiveBaseClass


ABC_METHOD_NOT_IMPL_MSG = "You should implement this!"


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
        model: LVQBaseClass,
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
        raise NotImplementedError(ABC_METHOD_NOT_IMPL_MSG)


class _CallbackStop(Exception):
    """Stop at the accepted iterate, including on SciPy versions before 1.11."""


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
        gradient = self.objective.gradient(model, data, labels)
        normalize = getattr(model, "relevance_normalization", False)
        correction = getattr(model, "relevance_correction", None)
        if not normalize and correction is None:
            return gradient

        omega_gradient = model.to_omega(gradient)
        if normalize:
            trial = model.to_omega(variables)
            if correction is not None:
                trial = trial @ correction
            # Differentiate W / ||W|| separately for each trial relevance matrix.
            unit = model.omega_
            radial = np.sum(unit * omega_gradient, axis=(-2, -1), keepdims=True)
            omega_gradient -= unit * radial
            omega_gradient /= np.linalg.norm(trial, axis=(-2, -1), keepdims=True)
        if correction is not None:
            omega_gradient[:] = omega_gradient @ correction.T
        return gradient

    def solve(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        model: LVQBaseClass,
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

        if (callback := params.get("callback")) is not None:
            max_runs = (params.get("options") or {}).get("maxiter")
            if max_runs is None:
                # scipy.optimize.minimize defaults for BFGS and L-BFGS-B.
                max_runs = 200 * model.get_variables().size if self.method == "BFGS" else 15000

            def report(nit):
                return callback(_update_state(model, max_runs, (), nit=nit, fun=self.objective(model, data, labels)))

            if report("Initial"):
                return
            iteration = 0

            def on_iterate(xk):
                nonlocal iteration
                iteration += 1
                model.set_variables(xk)
                if report(iteration):
                    raise _CallbackStop

            params["callback"] = on_iterate

        try:
            result = spo.minimize(
                self._objective_wrapper,
                model.get_variables().copy(),
                method=self.method,
                args=(model, data, labels),
                **params,
            )
        except _CallbackStop:
            return

        # Update model
        model.set_variables(result.x)


def _update_state(model: LVQBaseClass, max_runs: int, state_keys, **kwargs: Any) -> dict:
    """Build a callback snapshot, leaving unavailable solver diagnostics as None."""
    state = dict.fromkeys(state_keys)
    state.update(model_params=model.get_model_params_snapshot(), max_runs=max_runs)
    state.update({key: value.copy() if isinstance(value, np.ndarray) else value for key, value in kwargs.items()})
    return state
