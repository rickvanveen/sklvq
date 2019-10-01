from . import ActivationBaseClass

import numpy as np


class SoftPlus(ActivationBaseClass):

    def __init__(self, beta=1):
        self.beta = beta

    def __call__(self, x):
        return np.log(1 + np.exp(self.beta * x))

    def gradient(self, x):
        exp = np.exp(self.beta * x)
        return (self.beta * exp) / (1 + exp)
