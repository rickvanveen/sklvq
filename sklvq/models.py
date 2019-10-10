from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

# Can be switched out by parameters to the models.
from . import activations
from . import discriminants
from . import distances
from . import solvers

# Cannot be switched out by parameters to the models.
from .objectives import GeneralizedLearning
from .objectives import GeneralizedMatrixLearning


def _conditional_mean(p_labels, data, d_labels):
    """ Implements the conditional mean, i.e., mean per class"""
    return np.array([np.mean(data[p_label == d_labels, :], axis=0)
                     for p_label in p_labels])


# Template (Context)
class LVQClassifier(ABC, BaseEstimator, ClassifierMixin):

    # Sklearn: You cannot change the value of the properties given in init.
    def __init__(self, distance_type, distance_params, solver_type, solver_params, prototypes_per_class, random_state):
        self.distance_type = distance_type
        self.distance_params = distance_params
        self.solver_type = solver_type
        self.solver_params = solver_params
        self.prototypes_per_class = prototypes_per_class
        self.random_state = random_state

    @abstractmethod
    def initialize(self, data, y):
        raise NotImplementedError("You should implement this! Must accept (data, y)"
                                  " and return Solver object")

    @abstractmethod
    def set(self, *args):
        raise NotImplementedError("You should implement this! Must accept (data, y)"
                                  " and return Solver object")

    @abstractmethod
    def get(self):
        raise NotImplementedError("You should implement this! Must accept (data, y)"
                                  " and return Solver object")

    @abstractmethod
    def set_variables(self, variables):
        raise NotImplementedError("You should implement this! Must accept (data, y)"
                                  " and return Solver object")

    @abstractmethod
    def get_variables(self):
        raise NotImplementedError("You should implement this! Must accept (data, y)"
                                  " and return Solver object")

    @abstractmethod
    def from_variables(self, variables):
        raise NotImplementedError("You should implement this! Must accept (data, y)"
                                  " and return Solver object")

    @staticmethod
    @abstractmethod
    def to_variables(*args):
        raise NotImplementedError("You should implement this! Must accept (data, y)"
                                  " and return Solver object")

    @abstractmethod
    def update(self, gradient):
        raise NotImplementedError("You should implement this! Must accept gradient update"
                                  " and modify all the variables that need updating")

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

    def fit(self, data, y):
        # Validate SciKit-learn required stuff... labels are indices for self.classes_ which contains the class labels.
        data, labels = self._validate(data, y)

        # SciKit-learn way of doing random stuff...
        self.random_state_ = check_random_state(self.random_state)

        # SciKit-learn list of checked params before predict
        self._to_be_checked_if_fitted = ['prototypes_', 'prototypes_labels_', 'classes_', 'random_state_']

        # Common LVQ steps
        # TODO: figure out where to put this logically... either like this or input to discriminant
        #  or objective
        self.distance_ = distances.grab(self.distance_type, self.distance_params)

        # I guess it's save to say that LVQ always needs to have initialized prototypes/prototype_labels
        if np.isscalar(self.prototypes_per_class):
            self.prototypes_labels_ = np.repeat(unique_labels(labels), self.prototypes_per_class)

        self.prototypes_ = self.init_prototypes(data, labels)
        # Initialize algorithm specific stuff
        objective = self.initialize(data, labels)

        solver = solvers.grab(self.solver_type, self.solver_params)

        return solver.solve(data, labels, objective, self)

    def predict(self, data):
        # Append in specific classifier
        check_is_fitted(self, self._to_be_checked_if_fitted)

        # Input validation
        data = check_array(data)

        # TODO: Reject option?
        # Prototypes labels are indices of classes_
        return self.prototypes_labels_.take(self.distance_(data, self).argmin(axis=1))


# Template (Context Implementation)
class GLVQClassifier(LVQClassifier):

    # NOTE: Objective should be fixed. If another objective is needed a new classifier should be created.
    def __init__(self,
                 distance_type='squared-euclidean', distance_params=None,
                 activation_type='identity', activation_params=None,
                 discriminant_type='relative-distance', discriminant_params=None,
                 solver_type='bgd', solver_params=None,
                 prototypes_per_class=1, random_state=None):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params

        super(GLVQClassifier, self).__init__(distance_type, distance_params,
                                             solver_type, solver_params,
                                             prototypes_per_class, random_state)

    def initialize(self, data, labels):
        """ . """
        self.variables_size_ = self.prototypes_.size

        activation = activations.grab(self.activation_type, self.activation_params)

        discriminant = discriminants.grab(self.discriminant_type, self.discriminant_params)

        objective = GeneralizedLearning(activation=activation, discriminant=discriminant)

        return objective

    # NOTE: not very interesting for GLVQ, but potentially useful for others.
    def set(self, prototypes):
        self.prototypes_ = prototypes

    def get(self):
        return self.prototypes_

    # TODO might be added to LVQBaseClass
    def set_variables(self, variables):
        self.set(self.from_variables(variables))

    # TODO might be added to LVQBaseClass
    def get_variables(self):
        return self.to_variables(self.get())

    def from_variables(self, variables):
        return variables.reshape(self.prototypes_.shape)

    @staticmethod
    def to_variables(prototypes):
        return prototypes.ravel()

    def update(self, gradient_update_variables):
        self.prototypes_ -= self.from_variables(gradient_update_variables)


class GMLVQClassifier(LVQClassifier):

    def __init__(self,
                 distance_type='adaptive-squared-euclidean', distance_params=None,
                 activation_type='identity', activation_params=None,
                 discriminant_type='relative-distance', discriminant_params=None,
                 solver_type='bgd', solver_params=None,
                 verbose=False,
                 prototypes_per_class=1, random_state=None):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params
        self.verbose = verbose

        super(GMLVQClassifier, self).__init__(distance_type, distance_params,
                                              solver_type, solver_params,
                                              prototypes_per_class, random_state)

    def initialize(self, data, labels):
        """ . """
        # Initialize omega.
        self.omega_ = self._normalise(np.diag(np.ones(data.shape[1])))

        # Depends also on local (per class/prototype) global omega # TODO: implement local per class and prototype
        self.variables_size_ = self.prototypes_.size + self.omega_.size

        activation = activations.grab(self.activation_type, self.activation_params)

        discriminant = discriminants.grab(self.discriminant_type, self.discriminant_params)

        objective = GeneralizedMatrixLearning(activation=activation, discriminant=discriminant)

        return objective

    def set(self, prototypes, omega):
        self.prototypes_ = prototypes
        self.omega_ = omega

    def get(self):
        return self.prototypes_, self.omega_

    # TODO might be added to LVQBaseClass
    def set_variables(self, variables):
        self.set(*self.from_variables(variables))

    # TODO might be added to LVQBaseClass
    def get_variables(self):
        return self.to_variables(*self.get())

    def from_variables(self, variables):
        prototypes = np.reshape(variables[:self.prototypes_.size], self.prototypes_.shape)
        omega = self._normalise(np.reshape(variables[self.prototypes_.size:], self.omega_.shape))
        return prototypes, omega

    @staticmethod
    def to_variables(prototypes, omega):
        return np.append(prototypes.ravel(), omega.ravel())

    def update(self, gradient_update_variables):
        self.set_variables(self.to_variables(*self.get()) - gradient_update_variables)

    @staticmethod
    def _normalise(omega):
        return omega / np.sqrt(np.sum(np.diagonal(omega.T.dot(omega))))
