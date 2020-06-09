from . import LVQBaseClass

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import TransformerMixin

from sklvq import activations, discriminants, objectives
from sklvq.objectives import GeneralizedLearningObjective

from typing import Tuple

ModelParamsType = Tuple[np.ndarray, np.ndarray]


# TODO: Could use different step-sizes for matrices
class LGMLVQ(LVQBaseClass, TransformerMixin):
    def __init__(
        self,
        distance_type="local-adaptive-squared-euclidean",
        distance_params=None,
        activation_type="identity",
        activation_params=None,
        discriminant_type="relative-distance",
        discriminant_params=None,
        solver_type="steepest-gradient-descent",
        solver_params=None,
        localization="p",  # p (prototype), c (class)
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
        self.localization = localization
        self.verbose = verbose

        super(LGMLVQ, self).__init__(
            distance_type,
            distance_params,
            solver_type,
            solver_params,
            prototypes_per_class,
            prototypes,
            random_state,
        )

    # Should be private (cannot be used without calling fit of LVQBaseClass)
    def initialize(self, data, y):
        """ . """
        # Initialize omega
        if self.omega is None:
            if self.localization == "p":
                self._num_omega = self.prototypes_.shape[0]
                # self._omega_labels = np.arange(self.prototypes_.size) # Corresponding to index of prototype
            elif self.localization == "c":
                self._num_omega = self.classes_.size
                # self._omega_labels = np.arange(self.classes_.size)
            self.omega_ = np.array(
                [np.eye(data.shape[1]) for _ in range(self._num_omega)]
            )
        else:
            self.omega_ = self.omega

        # Add some check
        self.omega_ = self._normalise_omega(self.omega_)

        activation = activations.grab(self.activation_type, self.activation_params)

        discriminant = discriminants.grab(
            self.discriminant_type, self.discriminant_params
        )

        objective = GeneralizedLearningObjective(
            activation=activation, discriminant=discriminant
        )

        return objective

    def set_model_params(self, model_params):
        (self.prototypes_, omega) = model_params
        self.omega_ = self._normalise_omega(omega)
        self.omega_ = omega

    def get_model_params(self):
        return self.prototypes_, self.omega_

    def to_params(self, variables):
        # Return tuple of correctly reshaped prototypes and omegas
        return (
            np.reshape(variables[0 : self.prototypes_.size], self.prototypes_.shape),
            np.reshape(variables[self.prototypes_.size :], self.omega_.shape),
        )

    def to_variables(self, model_params: ModelParamsType) -> np.ndarray:
        omega_size = self.omega_.size
        prototypes_size = self.prototypes_.size

        variables = np.zeros(prototypes_size + omega_size)

        (variables[0:prototypes_size], variables[prototypes_size:]) = map(
            np.ravel, model_params
        )

        return variables

    def normalize_params(self, model_params: ModelParamsType) -> ModelParamsType:
        (prototypes, omega) = model_params
        normalized_prototypes = prototypes / np.linalg.norm(
            prototypes, axis=1, keepdims=True
        )
        normalized_omega = LGMLVQ._normalise_omega(omega)
        return (normalized_prototypes, normalized_omega)

    @staticmethod #TODO: switch off normalization... as option, normalize all of them or none.
    def _normalise_omega(omega: np.ndarray) -> np.ndarray:
        denominator = np.sqrt(np.einsum("ijk,ijk->i", omega, omega)).reshape(
            omega.shape[0], 1, 1
        )
        # denominator[denominator == 0] = 1.0
        return omega / denominator

    # def _get_tags(self):
    #     return {"poor_performance"}