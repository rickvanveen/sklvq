from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state, shuffle
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets

import numpy as np

from . import distances
from . import activations
from lvqtoolbox.objectives import relative_distance_objective


def _conditional_mean(p_labels, data, d_labels):
    """ Implements the conditional mean, i.e., mean per class"""
    return np.array([np.mean(data[p_label == d_labels, :], axis=0)
                     for p_label in p_labels])


# Template (Context)
class LVQClassifier(ABC, BaseEstimator, ClassifierMixin):

    # But here I can have logic... because it's not a sklearn estimator?
    # Cannot change the value of the properties given in init...
    def __init__(self, prototypes_per_class, random_state):
        self.prototypes_per_class = prototypes_per_class
        self.random_state = random_state

    @abstractmethod
    def initialize(self, data, y):
        raise NotImplementedError("You should implement this! Must accept (data, y)"
                                  " and return Objective and Solver objects")

    # TODO: could also be class functions things that can be extended by user by providing another class same for omega.
    def init_prototypes(self, data, y):
        conditional_mean = _conditional_mean(self.prototypes_labels_, data, y)
        return conditional_mean + (1e-4 * self.random_state_.uniform(-1, 1, conditional_mean.shape))

    def _validate(self, data, y):
        # SciKit-learn required check
        data, labels = check_X_y(data, y)

        # SciKit-learn required check
        check_classification_targets(y)

        # Scikit-learn requires classes_ which stores the labels, y is now an array of indices from classes_.
        self.classes_, y = np.unique(y, return_inverse=True)

        return data, labels

    # TODO: Change to _fit_solver1 and _fit_solver2 and call in the specific the right fit
    def fit(self, data, y):
        # Validate SciKit-learn required stuff... labels are indices for self.classes_ which contains the class labels.
        data, labels = self._validate(data, y)

        # SciKit-learn way of doing random stuff...
        self.random_state_ = check_random_state(self.random_state)

        # SciKit-learn list of checked params before predict
        self._to_be_checked_if_fitted = ['prototypes_', 'prototypes_labels_', 'classes_', 'random_state_']

        # Common LVQ steps
        # I guess it's save to say that LVQ always needs to have initialized prototypes/prototype_labels
        if np.isscalar(self.prototypes_per_class):
            self.prototypes_labels_ = np.repeat(unique_labels(labels), self.prototypes_per_class)

        self.prototypes_ = self.init_prototypes(data, labels)
        # Initialize algorithm specific stuff
        solver = self.initialize(data, labels)

        # Should be done by subclass...
        # self.restore_from_variables(variables)

        return solver(data, labels)

    def predict(self, data):
        # Append in specific classifier
        check_is_fitted(self, self._to_be_checked_if_fitted)

        # Input validation
        data = check_array(data)

        # Prototypes labels are indices of classes_
        return self.prototypes_labels_.take(self._distancefun(data, self.prototypes_).argmin(axis=1))


# Template (Context Implementation)
class GLVQClassifier(LVQClassifier):

    # TODO: The only way to support multiple algorithms with different parameters is to accept the whole list of
    #  parameters and set them to None by default...
    def __init__(self, distance='sqeuclidean', distance_params=None, activation='identity', activation_params=None, solver='l-bfgs-b', verbose=False,
                 prototypes_per_class=1, random_state=None):
        self.distance = distance
        self.distance_params = distance_params
        self.activation = activation
        self.activation_params = activation_params
        self.solver = solver
        self.verbose = verbose
        super(GLVQClassifier, self).__init__(prototypes_per_class, random_state)

    def initialize(self, data, labels):
        """ . """

        # Probably necessary to be available later
        # TODO: Probably needs to be done differently as predict can
        #  depend on more then just distancefun e.g., reject option?
        self._distancefun = distances.create(self.distance, self.distance_params)

        # Initialisation of Objective function
        activation = activations.create(self.activation, self.activation_params)

        objective = relative_distance_objective.RelativeDistanceObjective(self._distancefun, activation)

        # Initialisation of Solver ['sgd', 'bgd']
        solver = StochasticGradientDescent(objective=objective, model=self)

        return solver
        # , variables, objective_args

    # def restore_from_variables(self, variables):
    #     self.prototypes_ = variables.reshape(self.prototypes_.shape)


def _find_min(indices, distances):
    """ Helper function to find the minimum distance and the index of this distance. """
    dist_temp = np.where(indices, distances, np.inf)
    return dist_temp.min(axis=1), dist_temp.argmin(axis=1)


class RelativeDistanceObjective:
    def __init__(self, distance=None, activation=None):
        self.distance = distance
        self.activation = activation

    @staticmethod
    def _relative_distance(dist_same, dist_diff):
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
            return (dist_same - dist_diff) / (dist_same + dist_diff)

    @staticmethod
    def _relative_distance_gradient(dist_same, dist_diff):
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
            return 2 * dist_diff / (dist_same + dist_diff) ** 2

    def gradient(self, data, labels, prototypes, prototype_labels):
        distances = self.distance(data, prototypes)

        ii_same = np.transpose(np.array([labels == prototype_label for prototype_label in prototype_labels]))
        ii_diff = ~ii_same

        dist_same, i_dist_same = _find_min(ii_same, distances)
        dist_diff, i_dist_diff = _find_min(ii_diff, distances)

        relative_distance = self._relative_distance(dist_same, dist_diff)

        gradient = np.zeros(prototypes.shape)

        for i_prototype in range(0, prototype_labels.size):
            # Find for which samples this prototype is the closest and has the same label
            ii_same = (i_prototype == i_dist_same)
            if any(ii_same):
                # (dS / dmu * dmu / dd_J * dd_J / dw_J) J closest with same label
                magnitude = (self.activation.gradient(relative_distance[ii_same]) *
                             self._relative_distance_gradient(dist_same[ii_same], dist_diff[ii_same]))
                direction = data[ii_same, :] - prototypes[i_prototype, :]

                gradient[i_prototype, :] += (magnitude.dot(direction)).squeeze()

            # Find for which samples this prototype is the closest and has a different label
            ii_diff = (i_prototype == i_dist_diff)
            if any(ii_diff):
                # (dS / dmu * dmu / dd_K * dd_K / dw_K) K closest with different label
                magnitude = (self.activation.gradient(relative_distance[ii_diff]) *
                             self._relative_distance_gradient(dist_diff[ii_diff], dist_same[ii_diff]))
                direction = data[ii_diff] - prototypes[i_prototype, :]

                gradient[i_prototype, :] -= (magnitude.dot(direction)).squeeze()

        return gradient


# TODO: GradientDescent abstract super class with Stochastic, Batch, Mini-Batch subclasses
# TODO: convergence check... gradient zero?
# TODO: Anealing strategies

class StochasticGradientDescent:

    def __init__(self, objective=None, model=None, max_runs=5, batch_size=150, step_size=0.05):
        self.objective = objective
        self.model = model
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
