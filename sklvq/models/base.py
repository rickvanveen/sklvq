from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

# Can be switched out by parameters to the models.
from sklvq import distances, solvers, discriminants


from collections.abc import Iterable


def _conditional_mean(p_labels, data, d_labels):
    """ Implements the conditional mean, i.e., mean per class"""
    return np.array([np.mean(data[p_label == d_labels, :], axis=0)
                     for p_label in p_labels])


class LVQClassifier(ABC, BaseEstimator, ClassifierMixin):

    # Sklearn: You cannot change the value of the properties given in init.
    def __init__(self, distance_type, distance_params, solver_type, solver_params,
                 prototypes_per_class, prototypes, random_state):
        self.distance_type = distance_type
        self.distance_params = distance_params
        self.solver_type = solver_type
        self.solver_params = solver_params
        self.prototypes_per_class = prototypes_per_class
        self.prototypes = prototypes
        self.random_state = random_state

    @abstractmethod
    def initialize(self, data, y):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def set_model_params(self, model_params):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def get_model_params(self):
        raise NotImplementedError("You should implement this!")

    @staticmethod
    def to_variables(model_params):
        return np.concatenate(list(map(np.ravel, model_params)))
        # return np.concatenate(model_params).ravel()

    @abstractmethod
    def to_params(self, variables):
        raise NotImplementedError("You should implement this!")

    @staticmethod
    @abstractmethod
    def normalize_params(model_params):
        raise NotImplementedError("You should implement this!")

    @staticmethod
    @abstractmethod
    def mul_params(model_params, other):
        raise NotImplementedError("You should implement this!")

    @staticmethod
    def normalize_prototypes(prototypes):
        return prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)

    # TODO: could also be class functions thing that can be extended by user by providing another class same for omega.
    def init_prototypes(self, data, y):
        conditional_mean = _conditional_mean(self.prototypes_labels_, data, y)
        return conditional_mean + (1e-4 * self.random_state_.uniform(-1, 1, conditional_mean.shape))

    def _validate(self, data, labels):
        # SciKit-learn required check
        data, labels = check_X_y(data, labels)

        # SciKit-learn required check
        check_classification_targets(labels)

        # Scikit-learn requires classes_ which stores the labels, y is now an array of indices from classes_.
        self.classes_, _ = np.unique(labels, return_inverse=True)

        return data, labels

    def fit(self, data, y):
        # Validate SciKit-learn required stuff... labels are indices for self.classes_ which contains the class labels.
        data, labels = self._validate(data, y)

        # SciKit-learn way of doing random stuff...
        self.random_state_ = check_random_state(self.random_state)

        # Common LVQ steps
        self.distance_ = distances.grab(self.distance_type, self.distance_params)

        # Always given by default value accepts anything
        if np.isscalar(self.prototypes_per_class):
            self.prototypes_labels_ = np.repeat(unique_labels(labels), self.prototypes_per_class)

        # If prototypes is not already set initialize them to the class conditional mean.
        if self.prototypes is None:
            self.prototypes_ = self.init_prototypes(data, labels)
        else:
            self.prototypes_ = self.prototypes

        # Initialize algorithm specific stuff
        objective = self.initialize(data, labels)

        solver = solvers.grab(self.solver_type, self.solver_params)

        return solver.solve(data, labels, objective, self)

    def decision_function(self, data):
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        data = check_array(data)

        # Of shape n_observations , n_prototypes
        distances = self.distance_(data, self)

        # Allocation
        min_distances = np.zeros((data.shape[0], self.classes_.size))

        # return n_observations, n_classes
        for i, c in enumerate(self.classes_):
            min_distances[:, i] = distances[:, self.prototypes_labels_ == c].min(axis=1)

        sum_min_distances = np.sum(min_distances, axis=1)

        return (1 - (min_distances / sum_min_distances[:, np.newaxis])) / 2

    def predict_proba(self, data):
        return self.decision_function(data)

    def predict(self, data):
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        data = check_array(data)

        decision_values = self.decision_function(data)

        # TODO: Reject option?
        # Prototypes labels are indices of classes_
        # return self.prototypes_labels_.take(self.distance_(data, self).argmin(axis=1))
        return self.classes_[decision_values.argmax(axis=1)]