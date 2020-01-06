from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass
from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklvq.models import LVQClassifier

# TODO: Not correct. Needs to be changed

#Adam *
class AdaptiveMomentEstimation(SolverBaseClass):

    def __init__(self, max_runs: int = 20, batch_size: int = 1):
        self.max_runs = max_runs
        self.batch_size = batch_size
        self.step_size = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.999

    def solve(self, data: np.ndarray,
              labels: np.ndarray,
              objective: ObjectiveBaseClass,
              model: 'LVQClassifier') -> 'LVQClassifier':
        # Note: time t is not reset per run, but continues on

        # Adam
        average_gradient = np.zeros(model.variables_size_)
        average_squared_gradient = np.zeros(model.variables_size_)

        # average_delta_update = np.zeros(model.variables_size_)

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
                # Select data to base the update on...
                batch = data[batches[i_batch], :]
                batch_labels = labels[batches[i_batch]]

                # Adadelta
                gradient = objective.gradient(
                    model.get_variables(),
                    model,
                    batch,
                    batch_labels)

                average_gradient = self.beta1 * average_gradient + (1 - self.beta1) * gradient
                average_squared_gradient = self.beta2 * average_squared_gradient + (1 - self.beta2) * gradient**2

                corrected_average_gradient = average_gradient / (1 - self.beta1)
                corrected_average_squared_gradient = average_squared_gradient / (1 - self.beta2)

                delta_update = ((self.step_size / (np.sqrt(corrected_average_squared_gradient) + 1e-8))
                                * corrected_average_gradient)

                model.update(delta_update)

        return model
