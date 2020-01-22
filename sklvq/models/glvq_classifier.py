from . import LVQClassifier
import inspect
from typing import Union

# Can be switched out by parameters to the models.
from sklvq import activations, discriminants

# Cannot be switched out by parameters to the models.
from sklvq.objectives import GeneralizedLearningObjective

# TODO: White list of methods suitable for GLVQ


# Template (Context Implementation)
class GLVQClassifier(LVQClassifier):

    # NOTE: Objective will be fixed. If another objective is needed a new classifier and objective should be created.
    def __init__(self,
                 distance_type='squared-euclidean', distance_params=None,
                 activation_type = 'identity', activation_params=None,
                 discriminant_type='relative-distance', discriminant_params=None,
                 solver_type='sgd', solver_params=None,
                 prototypes=None, prototypes_per_class=1, random_state=None):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params

        super(GLVQClassifier, self).__init__(distance_type, distance_params,
                                             solver_type, solver_params,
                                             prototypes_per_class, prototypes, random_state)

    def initialize(self, data, labels):
        """ . """
        self.variables_size_ = self.prototypes_.size

        activation = activations.grab(self.activation_type, self.activation_params)

        discriminant = discriminants.grab(self.discriminant_type, self.discriminant_params)

        objective = GeneralizedLearningObjective(activation=activation, discriminant=discriminant)

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