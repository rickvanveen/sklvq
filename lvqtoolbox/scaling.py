from abc import ABC, abstractmethod

import numpy as np


class AbstractScaling(ABC):

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, x):
        raise NotImplementedError("You should implement this!")


class Identity(AbstractScaling):

    def __call__(self, x):
        """ Implements the identity function: f(x) = x

        Note helps with single interface in cost function.

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        x : ndarray, same shape and values as input.
        """
        return x

    def gradient(self, x):
        """ Implements the identity function derivative: g(x) = 1

        Note helps with single interface in cost function.

        Parameters
        ----------
        x : Anything

        Returns
        -------
        gradient : scalar,
                   Returns the constant 1 no matter the shape or values of the input.
        """
        return 1


class Sigmoid(AbstractScaling):

    def __init__(self, beta=2):
        self.beta = beta

    def __call__(self, x):
        """ Implements the sigmoid function: f(x) = 1 /( e^{-beta * x} + 1)

        Parameters
        ----------
        x    : ndarray
               The values that need to be scaled.

        Returns
        -------
        scaled_x : ndarray, same shape as input
                   The elementwise scaled input values.
        """
        return 1 / (np.exp(-self.beta * x) + 1)

    def gradient(self, x):
        """ Implements the sigmoid function's derivative: g(x) = (beta * e^(beta * x)) / (e^(beta * x) + 1)^2

        Parameters
        ----------
         x    : ndarray
               The values that need to be scaled.

        Returns
        -------
        gradient : ndarray, same shape as input
                   The elementwise scaled input values
        """
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