from . import ActivationBaseClass
import numpy as np


class Swish(ActivationBaseClass):

    def __init__(self, beta=1):
        self.beta = beta

    def _sgd(self, x):
        return 1 / (np.exp(-self.beta * x) + 1)

    def _swish(self, x):
        return x * self._sgd(x)

    def __call__(self, x):
        return self._swish(x)

    def gradient(self, x):
        return (self.beta * self._swish(x)) + (self._sgd(x) * (1 - self.beta * self._swish(x)))
