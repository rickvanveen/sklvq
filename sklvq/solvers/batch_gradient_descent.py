from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass


# TODO: GradientDescent abstract super class with Stochastic, (Batch, Mini-Batch) subclasses How to divide...
# TODO: convergence check... gradient zero?
# TODO: Annealing strategies


class BatchGradientDescent(SolverBaseClass):

    def __init__(self, max_runs=12, batch_size=50, step_size=0.05):
        self.max_runs = max_runs
        self.batch_size = batch_size
        self.step_size = step_size

    def solve(self, data, labels, objective, model):
        for i_run in range(0, self.max_runs):
            shuffled_indices = shuffle(range(0, labels.size),
                                       random_state=model.random_state_)

            num_batches = int(np.ceil(labels.size / self.batch_size))
            batches = []
            for i_batch in range(0, num_batches):
                i_start = i_batch * self.batch_size
                i_end = i_start + self.batch_size

                if i_end > labels.size:
                    i_end = labels.size

                batches.append(shuffled_indices[i_start:i_end])

            for i_batch in range(0, num_batches):
                # Select data to base the update on...
                batch = data[batches[i_batch], :]
                batch_labels = labels[batches[i_batch]]

                gradient = objective.gradient(batch, batch_labels, model)
                # TODO: add more stuff, e.g., waypoint averaging, convergence check

                model.update(self.step_size * gradient)

        return model
