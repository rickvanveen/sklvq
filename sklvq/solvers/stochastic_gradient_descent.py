from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass


class StochasticGradientDescent(SolverBaseClass):

    def __init__(self, objective=None, max_runs=5, batch_size=150, step_size=0.05):
        self.objective = objective
        self.max_runs = max_runs
        self.batch_size = batch_size
        self.step_size = step_size





    def __call__(self, data, labels):
        for i_run in range(0, self.max_runs):
            shuffled_indices = shuffle(range(0, labels.size), random_state=self.model.random_state_)

            # TODO: Special case batch_size = 1 optimization
            # TODO: Special case batch_size = labels.size optimization
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
                # Update variables in context of glvq prototypes (give object or prototypes)
                gradient = self.objective.gradient(batch,
                                                   batch_labels,
                                                   self.model.prototypes_,
                                                   self.model.prototypes_labels_)

                # apply update (setter for prototypes)
                # self.GLVQClassifier.update(update)

                # TODO: Potentially this can be moved to the model itself as it has all the parameters for it and
                #  then this doesn't need to be changed for every algorithm?? model.update(gradient)
                self.model.prototypes_ += self.step_size * gradient

                # gradient = objective.gradient(batch, batch_labels, model)
                # s

                # return GLVQClassifier with updated prototypes etc.
        return self.model
