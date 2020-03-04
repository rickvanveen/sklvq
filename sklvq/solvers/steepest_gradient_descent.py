from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass
from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class SteepestGradientDescent(SolverBaseClass):

    def __init__(self, max_runs=10, batch_size=10, step_size=np.array(0.2)):
        self.max_runs = max_runs
        self.batch_size = batch_size
        self.step_size = step_size

    def solve(self, data: np.ndarray,
              labels: np.ndarray,
              objective: ObjectiveBaseClass,
              model: 'LVQClassifier') -> 'LVQClassifier':

        step_size = self.step_size
        for i_run in range(0, self.max_runs):

            shuffled_indices = shuffle(
                range(0, labels.size),
                random_state=model.random_state_)

            batches = np.array_split(
                shuffled_indices,
                list(range(self.batch_size,
                           labels.size,
                           self.batch_size)),
                axis=0)

            for i_batch in range(0, len(batches)):
                # Select the batch
                batch = data[batches[i_batch], :]
                batch_labels = labels[batches[i_batch]]

                # Get model params variable shape (flattened)
                model_variables = model.to_variables(
                    model.get_model_params()
                )

                # Compute the objective gradient
                objective_gradient = model.to_params(
                    objective.gradient(
                        model_variables,
                        model,
                        batch,
                        batch_labels
                    )
                )

                # Apply the step size to the model parameters
                objective_gradient = model.to_variables(
                    model.mul_params(
                        objective_gradient,
                        step_size
                    )
                )

                # Update the model
                model.set_model_params(
                    model.to_params(
                        model_variables - objective_gradient
                    )
                )

            step_size = self.step_size / (1 + i_run/self.max_runs)

        return model
