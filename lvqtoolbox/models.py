from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets

import numpy as np

from lvqtoolbox.distance import DistanceFactory
from lvqtoolbox.scaling import ScalingFactory
from lvqtoolbox.solvers import SolverFactory

from lvqtoolbox.objective import RelativeDistanceObjective
from lvqtoolbox.objective import RelevanceRelativeDistanceObjective


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

    # TODO: Could move this to objective... and "make" it publicly availble objective is tightly coupled with the
    # TODO: algorithms anyway + we also need these functions in the objective...
    @abstractmethod
    def restore_from_variables(self, variables):
        raise NotImplementedError("You should implement this! Must accept variables"
                                  " and return correctly shaped variables depending on algorithm")

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
        solver, variables, objective_args = self.initialize(data, labels)

        variables = solver.solve(variables, objective_args)

        # Should be done by subclass...
        self.restore_from_variables(variables)

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
        # Here add inits for RelativeDistanceObjective... self.distance, self.scaling...
        self._distancefun = DistanceFactory.create(self.distance)
        scaling = ScalingFactory.create(self.scaling)
        scaling.beta = self.beta

        objective = RelativeDistanceObjective(distance=self._distancefun, scaling=scaling,
                                              prototypes_shape=self.prototypes_.shape)

        solver = SolverFactory.create(self.solver)
        solver.objective = objective

        variables = self.prototypes_.ravel()
        objective_args = (self.prototypes_labels_, data, labels)

        return solver, variables, objective_args

    def restore_from_variables(self, variables):
        self.prototypes_ = variables.reshape(self.prototypes_.shape)


class GMLVQClassifier(LVQClassifier):

    def __init__(self, distance='rel-sqeuclidean', solver='l-bfgs-b', scaling='identity', beta=None, verbose=False,
                 omega='identity', omega_shape=None,  prototypes_per_class=1, random_state=None):
        self.distance = distance
        self.solver = solver
        self.scaling = scaling
        self.beta = beta
        self.verbose = verbose
        self.omega = omega
        self.omega_shape = omega_shape
        super(GMLVQClassifier, self).__init__(prototypes_per_class, random_state)

    def _init_relevance(self, data):
        num_features = data.shape[1]

        # Random/ identity/ some other... maybe also objects? and able to extend them...
        if not self.omega_shape: # can only be square (identity) else use eye
            return np.identity(num_features) / num_features

    def initialize(self, data, labels):
        self._distancefun = DistanceFactory.create(self.distance)
        scaling = ScalingFactory.create(self.scaling)
        scaling.beta = self.beta

        # Initialise omega
        self.omega_ = self._init_relevance(data)

        # TODO: self.objective? or store final distance... as self.distance (need different name then in init)
        objective = RelevanceRelativeDistanceObjective(distance=self._distancefun, scaling=scaling,
                                                       prototypes_shape=self.prototypes_.shape,
                                                       omega_shape=self.omega_.shape)
        solver = SolverFactory.create(self.solver)
        solver.objective = objective

        # Construct variables array
        variables = self.prototypes_.ravel()
        variables = np.append(variables, self.omega_.ravel())

        objective_args = (self.prototypes_labels_, data, labels)

        return solver, variables, objective_args

    def restore_from_variables(self, variables):
        prototypes_variables = variables[:self.prototypes_.size]
        self.prototypes_ = prototypes_variables.reshape(self.prototypes_.shape)

        omega_variables = variables[self.prototypes_.size:]
        self.omega_ = omega_variables.reshape(self.omega_.shape)
        self.omega_ = self.omega_ / np.sqrt(np.sum(np.diagonal(self.omega_.T.dot(self.omega_))))

        self._distancefun.omega = self.omega_
