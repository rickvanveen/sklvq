from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass


# TODO: GradientDescent abstract super class with Stochastic, (Batch, Mini-Batch) subclasses How to divide...
# TODO: MORE solver schemes
# TODO: convergence check... gradient zero?
# TODO: Annealing strategies/functions
# TODO: statistics.

class SteepestGradientDescent(SolverBaseClass):

    def __init__(self, max_runs=10, batch_size=25, step_size=0.05):
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

                # gradient = objective.gradient(batch, batch_labels, model)
                gradient = objective.gradient(model.get_variables(), model, batch, batch_labels)
                # TODO: add more stuff, e.g., waypoint averaging

                model.update(self.step_size * gradient)

                # TODO: Here we need to report statistics like Cost etc...

                # TODO: Convergence check... basically choose some appropriate threshold in terms of gradient change
                #  or cost... Cost is better it might just go from one side of the minimum to the other which doesn't
                #  mean a small gradient but does mean no change in cost?

        return model

