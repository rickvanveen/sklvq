import numpy as np

from . import ActivationBaseClass


class Sigmoid(ActivationBaseClass):

    def __init__(self, beta=1):
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
