from . import LVQClassifier

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array

from sklvq import activations, discriminants, objectives
from sklvq.objectives import GeneralizedLearningObjective

from typing import Tuple

ModelParamsType = Tuple[np.ndarray, np.ndarray]


# TODO: Could use different step-sizes for matrices
class LGMLVQClassifier(LVQClassifier):
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

        super(LGMLVQClassifier, self).__init__(
            distance_type,
            distance_params,
            solver_type,
            solver_params,
            prototypes_per_class,
            prototypes,
            random_state,
        )

    # Should be private (cannot be used without calling fit of LVQClassifier)
    def initialize(self, data, y):
        """ . """

        # Initialize omega
        if self.omega is None:
            if self.localization == "p":
                self._num_omega = self.prototypes_.shape[0]
            elif self.localization == "c":
                self._num_classes = self.classes_.size
            self.omega_ = np.array(
                [np.eye(data.shape[1]) for _ in range(self._num_omega)]
            )
        else:
            self.omega_ = self.omega

        # Add some check
        self.omega_ = np.array([self._normalise_omega(omega) for omega in self.omega_])

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
        self.omega_ = np.array([self._normalise_omega(omega) for omega in self.omega_])

    def get_model_params(self):
        return self.prototypes_, self.omega_

    def to_params(self, variables):
        # First part of the variables are the prototypes
        prototype_indices = range(self.prototypes_.size)

        # Second part are the omegas
        omega_indices = range(self.prototypes_.size, variables.size)

        # Return tuple of correctly reshaped prototypes and omegas
        return (
            np.reshape(np.take(variables, prototype_indices), self.prototypes_.shape),
            np.reshape(np.take(variables, omega_indices), self.omega_.shape),
        )

    @staticmethod
    def normalize_params(model_params: ModelParamsType) -> ModelParamsType:
        (prototypes, omega) = model_params
        normalized_prototypes = prototypes / np.linalg.norm(
            prototypes, axis=1, keepdims=True
        )
        normalized_omega = np.array([LGMLVQClassifier._normalise_omega(o) for o in omega])
        return (normalized_prototypes, normalized_omega)

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

    @staticmethod
    def _normalise_omega(omega: np.ndarray) -> np.ndarray:
        return omega / np.sqrt(np.sum(np.diagonal(omega.T.dot(omega))))
