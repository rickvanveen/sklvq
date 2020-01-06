from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass
from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklvq.models import LVQClassifier

# TODO: convergence check... gradient zero?
# TODO: Annealing strategies/functions (step, linear, cosine, restart)
# TODO: statistics. Callable like with scipy solvers so just one method...
# TODO: maximum batch size based on available memory and approximately required memory by algorithms.
# TODO: Early stopping, Way point averaging.
# TODO: Multiple step_sizes... e.g., different for prototypes and omega


class SteepestGradientDescent(SolverBaseClass):

    def __init__(self, max_runs=10, batch_size=10, step_size=0.05):
        self.max_runs = max_runs
        self.batch_size = batch_size
        self.step_size = step_size

    def solve(self, data: np.ndarray,
              labels: np.ndarray,
              objective: ObjectiveBaseClass,
              model: 'LVQClassifier') -> 'LVQClassifier':

        # step_size = self.step_size
        # max_t = self.max_runs * np.round(data.shape[0] / self.batch_size)

        # costs = np.zeros(9)
        for i_run in range(0, self.max_runs):

            # costs[i_run % 9] = objective(model.get_variables(), model, data, labels)

            # if (i_run >= 9) & (np.all(np.abs(np.diff(costs)) < 0.5)):
            #     print('\nSTOP after {} runs\n'.format(i_run))
            #     print(costs)
            #     return model

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

                gradient = objective.gradient(model.get_variables(), model, batch, batch_labels)

                # Spot for waypoint averaging?

                model.update(self.step_size * gradient)

                # Spot for changing step_size (linear)
                # step_size = step_size - (self.step_size / max_t)

        return model

