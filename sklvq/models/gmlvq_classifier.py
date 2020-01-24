from . import LVQClassifier
import inspect

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array

from sklvq import activations, discriminants, objectives
from sklvq.objectives import GeneralizedLearningObjective

from sklvq.objectives.generalized_learning_objective import _find_min


# TODO: Local variant
# TODO: Transform function sklearn


class GMLVQClassifier(LVQClassifier):

    def __init__(self,
                 distance_type='adaptive-squared-euclidean', distance_params=None,
                 activation_type='identity', activation_params=None,
                 discriminant_type='relative-distance', discriminant_params=None,
                 solver_type='sgd', solver_params=None, verbose=False,
                 prototypes=None, prototypes_per_class=1, omega=None, random_state=None):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params
        self.omega = omega
        self.verbose = verbose

        super(GMLVQClassifier, self).__init__(distance_type, distance_params,
                                              solver_type, solver_params,
                                              prototypes_per_class, prototypes, random_state)

    # Get's called in fit.
    def initialize(self, data, labels):
        """ . """
        # Initialize omega. TODO: Make dynamic like the rest.
        if self.omega is None:
            self.omega_ = np.diag(np.ones(data.shape[1]))
            self.omega_ = self._normalise(self.omega_)
        else:
            self.omega_ = self.omega

        # Depends also on local (per class/prototype) global omega # TODO: implement local per class and prototype
        self.variables_size_ = self.prototypes_.size + self.omega_.size

        activation = activations.grab(self.activation_type,
                                      self.activation_params)

        discriminant = discriminants.grab(self.discriminant_type,
                                          self.discriminant_params)

        objective = GeneralizedLearningObjective(activation=activation,
                                                 discriminant=discriminant)

        return objective

    def set_model_params(self, prototypes, omega):
        self.prototypes_ = prototypes
        self.omega_ = omega

    def get_model_params(self):
        return self.prototypes_, self.omega_

    def from_variables(self, variables):
        prototypes = np.reshape(variables[:self.prototypes_.size], self.prototypes_.shape)
        omega = self._normalise(np.reshape(variables[self.prototypes_.size:], self.omega_.shape))
        return prototypes, omega

    @staticmethod
    def to_variables(prototypes, omega):
        return np.append(prototypes.ravel(), omega.ravel())

    def update(self, gradient_update_variables):
        self.set_variables(self.to_variables(*self.get_model_params()) - gradient_update_variables)

    @staticmethod
    def _normalise(omega):
        return omega / np.sqrt(np.sum(np.diagonal(omega.T.dot(omega))))

    def fit_transform(self, data, y):
        return self.fit(data, y).transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        data = check_array(data)

        check_is_fitted(self)

        lambda_ = self.omega_.T.dot(self.omega_)
        # TODO: SVD for stability? and SVD flip for stable direction?
        eigvalues, eigenvectors = np.linalg.eig(lambda_)
        sorted_ii = np.argsort(eigvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_ii]

        # data_new = data.dot(np.sqrt(eigvalues) * eigenvectors)
        data_new = data.dot(eigenvectors)

        return data_new

    def dist_function(self, data):
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        data = check_array(data)

        distances = self.distance_(data, self)
        min_args = np.argsort(distances, axis=1)

        winner = distances[list(range(0, distances.shape[0])), min_args[:, 0]]
        runner_up = distances[list(range(0, distances.shape[0])), min_args[:, 1]]

        return np.abs(winner - runner_up) / (
                    2 * np.linalg.norm(self.prototypes_[min_args[:, 0], :] - self.prototypes_[min_args[:, 1], :]) ** 2)

    def rel_dist_function(self, data):
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        data = check_array(data)

        distances = np.sort(self.distance_(data, self))

        winner = distances[:, 0]
        runner_up = distances[:, 1]

        return (runner_up - winner) / (winner + runner_up)

    def d_plus_function(self, data):
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        data = check_array(data)

        distances = np.sort(self.distance_(data, self))

        winner = distances[:, 0]

        return -1 * winner
