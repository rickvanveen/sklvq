from abc import ABC, abstractmethod

import numpy as np


class AbstractScaling(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, *args, **kwargs):
        raise NotImplementedError("You should implement this!")


class Identity(AbstractScaling):

    def __call__(self, x):
        return x

    def gradient(self, x):
        return 1


class Sigmoid(AbstractScaling):

    def __init__(self, beta=2):
        self.beta = beta

    def __call__(self, x):
        return 1 / (np.exp(-self.beta * x) + 1)

    def gradient(self, x):
        return (self.beta * np.exp(self.beta * x)) / (np.exp(self.beta * x) + 1) ** 2


class ScalingFactory:

    @staticmethod
    def create(scaling_type):
        if scaling_type == 'identity':
            return Identity()
        if scaling_type == 'sigmoid':
            return Sigmoid()
        else:
            print("Distance type does not exist")