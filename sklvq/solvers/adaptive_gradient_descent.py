from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass
from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier

# TODO: Not correct. Needs to be changed

# AdaDelta
class AdaptiveGradientDescent(SolverBaseClass):

    def __init__(self, max_runs: int = 20, batch_size: int = 1):
        self.max_runs = max_runs
        self.batch_size = batch_size
        self.gamma = 0.9

    def solve(self, data: np.ndarray,
              labels: np.ndarray,
              objective: ObjectiveBaseClass,
              model: 'LVQClassifier') -> 'LVQClassifier':
        # Note: time t is not reset per run, but continues on

        # Adadelta
        average_gradient = np.zeros(model.variables_size_)
        average_delta_update = np.zeros(model.variables_size_)

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

                # E(g2)t
                average_gradient = self.gamma * average_gradient + (1 - self.gamma) * gradient ** 2

                # delta psi
                delta_update = ((np.sqrt(average_delta_update ** 2 + 1e-8))
                                / (np.sqrt(average_gradient ** 2 + 1e-8))) * gradient

                model.update(delta_update)

                # E(psi2)t - 1
                average_delta_update = (self.gamma * average_delta_update
                                        + (1 - self.gamma) * delta_update ** 2)

        return model
