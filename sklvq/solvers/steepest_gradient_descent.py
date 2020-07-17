from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass
from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass


# TODO: Smart initialization based on algorithm and solver

STATE_KEYS = ["variables", "nit", "fun", "jac", "step_size"]


class SteepestGradientDescent(SolverBaseClass):
    def __init__(
        self,
        objective: ObjectiveBaseClass,
        max_runs=10,
        batch_size=1,
        step_size=0.2,
        callback=None,
    ):
        super().__init__(objective)
        self.max_runs = max_runs
        self.batch_size = batch_size
        self.step_size = step_size
        self.callback = callback

    def solve(
        self, data: np.ndarray, labels: np.ndarray, model: "LVQBaseClass",
    ) -> "LVQBaseClass":

        if self.callback is not None:
            variables = model.to_variables(model.get_model_params())
            state = self.create_state(
                STATE_KEYS,
                variables=variables,
                nit=0,
                fun=self.objective(variables, model, data, labels),
            )
            if self.callback(model, state):
                return model

        objective_gradient = None

        for i_run in range(0, self.max_runs):
            # Randomize order of samples
            shuffled_indices = shuffle(
                range(0, labels.size), random_state=model.random_state_
            )

            batch_size = self.batch_size
            if batch_size <= 0:
                batch_size = data.shape[0]

            # Divide the shuffled indices into batches (not necessarily equal size,
            # see documentation of numpy.array_split). batch_size set to 1 equals the stochastic
            # variant
            batches = np.array_split(
                shuffled_indices,
                list(range(batch_size, labels.size, batch_size)),
                axis=0,
            )

            # Update step size using a simple annealing strategy
            step_size = self.step_size / (1 + i_run / self.max_runs)

            for i_batch in range(0, len(batches)):
                # Select the batch
                batch = data[batches[i_batch], :]
                batch_labels = labels[batches[i_batch]]

                # Get model params variable shape (flattened)
                model_variables = model.to_variables(model.get_model_params())

                # Transform the objective gradient to model_params form
                objective_gradient = model.to_params(
                    # Compute the objective gradient
                    self.objective.gradient(model_variables, model, batch, batch_labels)
                )

                # Transform objective gradient to variables form
                objective_gradient = model.to_variables(
                    # Apply the step size to the model parameters
                    self.multiply_model_params(step_size, objective_gradient)
                )

                # Update the model
                model.set_model_params(
                    # Subtract objective gradient of model params in variables form
                    # and transform back to parameters form.
                    model.to_params(model_variables - objective_gradient)
                )

            if self.callback is not None:
                variables = model.to_variables(model.get_model_params())
                state = self.create_state(
                    STATE_KEYS,
                    variables=variables,
                    nit=i_run,
                    fun=self.objective(variables, model, data, labels),
                    jac=objective_gradient,
                    step_size=step_size,
                )
                if self.callback(model, state):
                    return model

        return model
