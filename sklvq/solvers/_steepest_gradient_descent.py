import numpy as np
from sklearn.utils import shuffle

from . import SolverBaseClass
from ..objectives import ObjectiveBaseClass
from ._base import _update_state

from typing import Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass

STATE_KEYS = ["variables", "nit", "fun", "step_size"]


class SteepestGradientDescent(SolverBaseClass):
    """ SteepestGradientDescent

    Implements the stochastic, batch and mini-batch gradient descent optimization methods.

    Parameters
    ----------
    objective: ObjectiveBaseClass, required
        This is/should be set by the algorithm.
    max_runs: int
        Number of runs over all the X. Should be >= 1
    batch_size: int
        Controls the batch size. Use 1 for stochastic, 0 for all X (batch gradient descent),
        and any number > 1 for mini batch. For mini-batch the solver will do as many batches with
        the specified number as possible. The last batch may have less samples then specified.
    step_size: float or ndarray
        The step size to control the learning rate of the model parameters. If the same step_size
        should be used for all parameters (e.g., prototypes and omega) then a float is
        sufficient. If separate initial step_sizes should be used per model parameter then this
        should be specified by using a ndarray.
    callback: callable
        Callable with signature callable(state). If the callable returns True the solver
        will stop (early). The state object contains the following.

        - "variables"
            Concatenated 1D ndarray of the model's parameters
        - "nit"
            The current iteration counter
        - "fun"
            The objective cost
        - "step_size"
            The current step_size(s)

    """

    def __init__(
            self,
            objective: ObjectiveBaseClass,
            max_runs: int = 10,
            batch_size: int = 1,
            step_size: float = 0.2,
            callback: callable = None,
    ):
        super().__init__(objective)
        self.max_runs: int = max_runs
        self.batch_size: int = batch_size
        self.step_size: Union[float, np.ndarray] = step_size
        self.callback: callable = callback

    def solve(
            self, data: np.ndarray, labels: np.ndarray, model: "LVQBaseClass",
    ):
        """

        Parameters
        ----------
        data : ndarray of shape (number of observations, number of dimensions)
        labels : ndarray of size (number of observations)
        model : LVQBaseClass
            The initial model that will be changed and holds the results at the end

        """

        if self.callback is not None:
            state = _update_state(
                STATE_KEYS,
                variables=model.get_variables(),
                nit=0,
                fun=self.objective(model, data, labels),
            )
            if self.callback(state):
                return

        batch_size = self.batch_size

        if batch_size > data.shape[0]:
            raise ValueError("Provided batch_size is invalid.")

        if batch_size <= 0:
            batch_size = data.shape[0]

        for i_run in range(0, self.max_runs):
            # Randomize order of samples
            shuffled_indices = shuffle(
                np.array(range(0, labels.size)), random_state=model.random_state_
            )

            # Divide the shuffled indices into batches (not necessarily equal size,
            # see documentation of numpy.array_split).
            batches = np.array_split(
                shuffled_indices,
                list(range(batch_size, labels.size, batch_size)),
                axis=0,
            )

            # Update step size using a simple annealing strategy
            step_size = self.step_size / (1 + i_run / self.max_runs)

            for i_batch in batches:
                # Creating views when batch_size is 1 does not do much for speed.
                batch = data[i_batch, :]
                batch_labels = labels[i_batch]

                # Compute objective gradient
                objective_gradient = self.objective.gradient(model, batch, batch_labels)

                # Multiply each param by its given step_size
                model.multiply_variables(step_size, objective_gradient)

                # Update the model by subtracting the objective-gradient (descent) from the
                # current models variables, e.g., (prototypes, omega) in case of GMLVQ
                model.set_variables(
                    np.subtract(  # returns out=objective_gradient
                        model.get_variables(),
                        objective_gradient,
                        out=objective_gradient,
                    )
                )

            if self.callback is not None:
                state = _update_state(
                    STATE_KEYS,
                    variables=model.get_variables(),
                    nit=i_run + 1,
                    fun=self.objective(model, data, labels),
                    step_size=step_size,
                )
                if self.callback(state):
                    return
