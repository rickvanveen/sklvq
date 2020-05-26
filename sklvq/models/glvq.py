from . import LVQBaseClass
import numpy as np

# Can be switched out by parameters to the models.
from sklvq import activations, discriminants

# Cannot be switched out by parameters to the models.
from sklvq.objectives import GeneralizedLearningObjective

from typing import Tuple

ModelParamsType = np.ndarray


# Template (Context Implementation)
class GLVQ(LVQBaseClass):

    # NOTE: Objective will be fixed. If another objective is needed a new classifier and objective should be created.
    def __init__(
        self,
        distance_type="squared-euclidean",
        distance_params=None,
        activation_type="identity",
        activation_params=None,
        discriminant_type="relative-distance",
        discriminant_params=None,
        solver_type="steepest-gradient-descent",
        solver_params=None,
        prototypes=None,
        prototypes_per_class=1,
        random_state=None,
    ):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params

        super(GLVQ, self).__init__(
            distance_type,
            distance_params,
            solver_type,
            solver_params,
            prototypes_per_class,
            prototypes,
            random_state,
        )

    def initialize(self, data, labels):
        """ . """
        # Grab the chosen activation, and discriminant functions and initialize with parameters.
        activation = activations.grab(self.activation_type, self.activation_params)

        discriminant = discriminants.grab(
            self.discriminant_type, self.discriminant_params
        )

        # The objective is fixed as this determines what else to initialize.
        objective = GeneralizedLearningObjective(
            activation=activation, discriminant=discriminant
        )

        return objective

    # Functions used in optimizers...
    def set_model_params(self, model_params):
        self.prototypes_ = model_params

    def get_model_params(self):
        return self.prototypes_

    def to_params(self, variables):
        return np.reshape(variables, self.prototypes_.shape)

    def to_variables(self, model_params: ModelParamsType) -> np.ndarray:
        return model_params.ravel()

    @staticmethod
    def normalize_params(model_params):
        return LVQBaseClass.normalize_prototypes(model_params)

    @staticmethod
    def mul_params(
        model_params: ModelParamsType, other: Tuple[int, float]
    ) -> ModelParamsType:
        prots = model_params
        # Scalar int or float
        return prots * other
