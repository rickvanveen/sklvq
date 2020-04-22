from . import LVQClassifier

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array

from sklvq import activations, discriminants, objectives
from sklvq.objectives import GeneralizedLearningObjective

from typing import Tuple

ModelParamsType = Tuple[np.ndarray, np.ndarray]

# TODO: Local variant
# TODO: Transform function sklearn


class GMLVQClassifier(LVQClassifier):
    def __init__(
        self,
        distance_type="adaptive-squared-euclidean",
        distance_params=None,
        activation_type="identity",
        activation_params=None,
        discriminant_type="relative-distance",
        discriminant_params=None,
        solver_type="steepest-gradient-descent",
        solver_params=None,
        verbose=False,
        prototypes=None,
        prototypes_per_class=1,
        omega=None,
        random_state=None,
    ):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params
        self.omega = omega
        self.verbose = verbose

        super(GMLVQClassifier, self).__init__(
            distance_type,
            distance_params,
            solver_type,
            solver_params,
            prototypes_per_class,
            prototypes,
            random_state,
        )

    # Get's called in fit.
    def initialize(self, data, labels):
        """ . """
        if self.omega is None:
            self.omega_ = np.eye(data.shape[1])
        else:
            self.omega_ = self.omega

        self.omega_ = self._normalise_omega(self.omega_)

        # Depends also on local (per class/prototype) global omega # TODO: implement local per class and prototype
        self.variables_size_ = self.prototypes_.size + self.omega_.size

        activation = activations.grab(self.activation_type, self.activation_params)

        discriminant = discriminants.grab(
            self.discriminant_type, self.discriminant_params
        )

        objective = GeneralizedLearningObjective(
            activation=activation, discriminant=discriminant
        )

        return objective

    def set_model_params(self, model_params: ModelParamsType) -> None:
        (self.prototypes_, omega) = model_params
        self.omega_ = self._normalise_omega(omega)

    def get_model_params(self) -> ModelParamsType:
        return self.prototypes_, self.omega_

    def to_params(self, variables: np.ndarray) -> ModelParamsType:
        # First part of the variables are the prototypes
        # prototype_indices = range(self.prototypes_.size)

        # Second part are the omegas
        # omega_indices = range(self.prototypes_.size, variables.size)

        # Return tuple of correctly reshaped prototypes and omegas
        # return (
        #     np.reshape(np.take(variables, prototype_indices), self.prototypes_.shape),
        #     np.reshape(np.take(variables, omega_indices), self.omega_.shape),
        # )
        return (
            np.reshape(variables[0:self.prototypes_.size], self.prototypes_.shape),
            np.reshape(variables[self.prototypes_.size:], self.omega_.shape)
        )

    @staticmethod
    def _normalise_omega(omega: np.ndarray) -> np.ndarray:
        return omega / np.sqrt(np.einsum("ij, ij", omega, omega))
        # return omega / np.sqrt(np.sum(np.diagonal(omega.T.dot(omega))))

    @staticmethod
    def normalize_params(model_params: ModelParamsType) -> ModelParamsType:
        (prototypes, omega) = model_params
        normalized_prototypes = prototypes / np.linalg.norm(
            prototypes, axis=1, keepdims=True
        )
        normalized_omega = GMLVQClassifier._normalise_omega(omega)
        return (normalized_prototypes, normalized_omega)

    def fit_transform(self, data: np.ndarray, y: np.ndarray) -> np.ndarray:
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
            2
            * np.linalg.norm(
                self.prototypes_[min_args[:, 0], :]
                - self.prototypes_[min_args[:, 1], :]
            )
            ** 2
        )

    # def rel_dist_function(self, data):
    #     # SciKit-learn list of checked params before predict
    #     check_is_fitted(self)
    #
    #     # Input validation
    #     data = check_array(data)
    #
    #     distances = np.sort(self.distance_(data, self))
    #
    #     winner = distances[:, 0]
    #     runner_up = distances[:, 1]
    #
    #     return (runner_up - winner) / (winner + runner_up)
    #
    # def d_plus_function(self, data):
    #     # SciKit-learn list of checked params before predict
    #     check_is_fitted(self)
    #
    #     # Input validation
    #     data = check_array(data)
    #
    #     distances = np.sort(self.distance_(data, self))
    #
    #     winner = distances[:, 0]
    #
    #     return -1 * winner

    @staticmethod
    def mul_params(
        model_params: ModelParamsType, other: Tuple[int, float, np.ndarray]
    ) -> ModelParamsType:
        (prots, omegs) = model_params
        if isinstance(other, np.ndarray):
            if other.size >= 2:
                return (prots * other[0], omegs * other[1])
        # Scalar int or float
        return (prots * other, omegs * other)
