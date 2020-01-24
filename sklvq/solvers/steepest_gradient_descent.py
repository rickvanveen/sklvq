from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass
from sklvq.objectives import ObjectiveBaseClass

from operator import mul

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

        if self.step_size.size is not model._number_of_params:
            self.step_size = np.repeat(self.step_size, model._number_of_params)

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
                # Select data to base the update on...
                batch = data[batches[i_batch], :]
                batch_labels = labels[batches[i_batch]]

                gradient = model.from_variables(objective.gradient(model.get_variables(), model, batch, batch_labels))

                # Applies step_size[0] (multiplies) to first returned model_param, steps_size[1] to second etc.
                if isinstance(gradient, tuple):
                    gradient = tuple(map(mul, tuple(np.atleast_1d(step_size)), gradient))
                    model.update(*gradient)
                else:
                    gradient = step_size * gradient
                    model.update(gradient)

            step_size = self.step_size / (1 + i_run/self.max_runs)

        return model
