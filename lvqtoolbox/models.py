from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets

import numpy as np

from lvqtoolbox.distance import DistanceFactory
from lvqtoolbox.scaling import ScalingFactory


# from lvqtoolbox.solvers import SolverFactory

# from lvqtoolbox.objective import RelativeDistanceObjective


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

    # # TODO: Could move this to objective... and "make" it publicly available as objective is tightly coupled with the algorithms anyway + we also need these functions in the objective...
    # @abstractmethod
    # def restore_from_variables(self, variables):
    #     raise NotImplementedError("You should implement this! Must accept variables"
    #                               " and return correctly shaped variables depending on algorithm")

    # TODO: could also be class functions things that can be extended by user by providing another class same for omega.
    def init_prototypes(self, data, y):
        conditional_mean = _conditional_mean(self.prototypes_labels_, data, y)
        return conditional_mean + (1e-4 * self.random_state_.uniform(-1, 1, conditional_mean.shape))

    def validate(self, data, y):
        # SciKit-learn required check
        data, labels = check_X_y(data, y)

        # SciKit-learn required check
        check_classification_targets(y)

        # Scikit-learn requires classes_ which stores the labels, y is now an array of indeces fro classes_.
        self.classes_, y = np.unique(y, return_inverse=True)

        return data, labels

    # TODO: Change to _fit_solver1 and _fit_solver2 and call in the specific the right fit
    def fit(self, data, y):
        # Validate SciKit-learn required stuff... labels are indices for self.classes_ which contains the class labels.
        data, labels = self.validate(data, y)

        # SciKit-learn way of doing random stuff...
        self.random_state_ = check_random_state(self.random_state)

        # Common LVQ steps
        # I guess it's save to say that LVQ always needs to have initialized prototypes/prototype_labels
        if np.isscalar(self.prototypes_per_class):
            self.prototypes_labels_ = np.repeat(unique_labels(labels), self.prototypes_per_class)

        self.prototypes_ = self.init_prototypes(data, labels)
        # Initialize algorithm specific stuff
        solver = self.initialize(data, labels)

        self.prototypes_ = solver(data, labels, self.prototypes_, self.prototypes_labels_)

        # Should be done by subclass...
        # self.restore_from_variables(variables)

        return self

    def predict(self, data):
        # TODO: The array should be set in subclass or append the array in subclass
        check_is_fitted(self, ['prototypes_', 'prototypes_labels_', 'classes_', 'random_state_'])

        # Input validation
        data = check_array(data)


        # Prototypes labels are indices of classes_
        return self.prototypes_labels_.take(self._distancefun(data, self.prototypes_).argmin(axis=1))


# Template (Context Implementation)
class GLVQClassifier(LVQClassifier):

    def __init__(self, distance='sqeuclidean', solver='l-bfgs-b', scaling='identity', beta=None, verbose=False,
                 prototypes_per_class=1, random_state=None):
        self.distance = distance
        self.solver = solver
        self.scaling = scaling
        self.beta = beta
        self.verbose = verbose
        super(GLVQClassifier, self).__init__(prototypes_per_class, random_state)

    def initialize(self, data, labels):
        """ . """

        # Probably necessary to be available later
        self._distancefun = DistanceFactory.create(self.distance)

        # Initialisation of Objective function
        scaling = ScalingFactory.create(self.scaling)

        objective = RelativeDistanceObjective(self._distancefun, scaling)

        # Initialisation of Solver ['sgd', 'bgd']
        solver = StochasticGradientDescent(objective=objective)

        return solver
            # , variables, objective_args

    # def restore_from_variables(self, variables):
    #     self.prototypes_ = variables.reshape(self.prototypes_.shape)


class AbstractObjective(ABC):

    @abstractmethod
    def __call__(self, variables, prototype_labels, data, labels):
        raise NotImplementedError("You should implement this! Should return"
                                  " objective value, gradient (1D, like variables)")


def _find_min(indices, distances):
    """ Helper function to find the minimum distance and the index of this distance. """
    dist_temp = np.where(indices, distances, np.inf)
    return dist_temp.min(axis=1), dist_temp.argmin(axis=1)


def _relative_distance(dist_same, dist_diff):
    return (dist_same - dist_diff) / (dist_same + dist_diff)


def _relative_distance_grad(dist_same, dist_diff):
    return 2 * dist_diff / (dist_same + dist_diff) ** 2


class RelativeDistanceObjective(AbstractObjective):
    def __init__(self, distance=None, activation=None):
        self.distance = distance
        self.activation = activation

    # Computes all the distances such that we only need to do this once.
    def _compute_distance(self, data, labels, prototypes, prototype_labels):
        """ Computes the distances between each prototype and each observation and finds all indices where the shortest
        distance is that of the prototype with the same label and with a different label. """

        distances = self.distance(data, prototypes)

        ii_same = np.transpose(np.array([labels == prototype_label for prototype_label in prototype_labels]))
        ii_diff = ~ii_same

        dist_same, i_dist_same = _find_min(ii_same, distances)
        dist_diff, i_dist_diff = _find_min(ii_diff, distances)

        return dist_same, dist_diff, i_dist_same, i_dist_diff

    def _gradient_activation(self, relative_distance, dist_same, dist_diff):
        gradient_scaling = (  # All samples where the current prototype is the closest and has the same label
                self.activation.gradient(relative_distance) *
                _relative_distance_grad(dist_same, dist_diff))
        return np.atleast_2d(gradient_scaling)

    # (dS/dmu * dmu/dd_i * dd_i/dw_i) for i J and K
    def _prototype_gradient(self, gradient_scaling, data, prototype):
        # d'_J(x, w) for all x (samples) in data and the current prototype
        dist_grad_wrt_prototype = np.atleast_2d(self.distance.gradient(data, prototype))
        return gradient_scaling.dot(dist_grad_wrt_prototype)

    # dS/dw_J() = (dS/dmu * dmu/dd_J * dd_J/dw_J) - (dS/dmu * dmu/dd_K * dd_K/dw_K)
    def _gradient(self, dist_same, dist_diff, i_dist_same, i_dist_diff,
                  relative_distance, data, prototypes, prototype_labels):
        gradient = np.zeros(prototypes.shape)
        for i_prototype in range(0, prototype_labels.size):
            # Find for which samples this prototype is the closest and has the same label
            ii_same = i_prototype == i_dist_same
            if any(ii_same):
                gradient_activation = self._gradient_activation(relative_distance[ii_same], dist_same[ii_same],
                                                                dist_diff[ii_same])
                gradient[i_prototype] = (
                        gradient[i_prototype] + self._prototype_gradient(gradient_activation, data[ii_same, :],
                                                                         prototypes[i_prototype, :]))

            # Find for which samples this prototype is the closes and has a different label
            ii_diff = i_prototype == i_dist_diff
            if any(ii_diff):
                gradient_activation = self._gradient_activation(relative_distance[ii_diff], dist_diff[ii_diff],
                                                                dist_same[ii_diff])
                gradient[i_prototype] = (
                        gradient[i_prototype] - self._prototype_gradient(gradient_activation, data[ii_diff, :],
                                                                         prototypes[i_prototype, :]))
        return gradient

    # S() = sum(f(mu(x)))
    def _cost(self, relative_distance):
        return np.sum(self.activation(relative_distance))

    def __call__(self, data, labels, prototypes, prototype_labels,):
        # dist_same, d_J(X, w_J), contains the distances between all
        # samples X and the closest prototype with the same label (_J)
        # i_dist_same tells you the index of the prototype that was closest.

        # dist_diff, d_K(X, w_K), contains the distances between all
        # samples X and the closest prototype with a different label (_K)
        # i_dist_diff same size as dist_diff, contains indices corresponding to the label of the closest prototype
        dist_same, dist_diff, i_dist_same, i_dist_diff = self._compute_distance(data, labels, prototypes,
                                                                                prototype_labels)
        # mu(x)
        relative_distance = _relative_distance(dist_same, dist_diff)

        # First part is S = Sum(f(u(x))), Second part: gradient of all the prototypes
        return self._cost(relative_distance), self._gradient(dist_same, dist_diff, i_dist_same,
                                                             i_dist_diff, relative_distance,
                                                             data, prototypes, prototype_labels)


class StochasticGradientDescent:

    def __init__(self, objective=None, max_runs=100, step_size=0.1):
        self.objective = objective
        self.max_runs = max_runs
        self.step_size = step_size

    # TODO: prototypes and other model related stuff could be send as a GLVQ object.
    def __call__(self, data, labels, prototypes, prototype_labels):
        for i_run in range(0, self.max_runs):
            cost = 0
            for i_data in range(0, labels.size):
                # Select data to base the update on...
                sample = np.atleast_2d(data[i_data, :])
                sample_label = labels[i_data]
                # Update variables in context of glvq prototypes (give object or prototypes)
                cost_update, prototype_update = self.objective(sample, sample_label, prototypes, prototype_labels)

                cost += cost_update
                # apply update (setter for prototypes)
                # self.GLVQClassifier.update(update)
                prototypes += self.step_size * prototype_update

                # return GLVQClassifier with updated prototypes etc.
        return prototypes
