import scipy as sp
import numpy as np

from . import DistanceBaseClass


class CosineDistance(DistanceBaseClass):

    def __init__(self, beta=1):
        self.beta = beta

    def _gamma(self, x):
        return (np.exp(-self.beta * (x - 1)) - 1) / (np.exp(2 * self.beta) - 1)

    def _gamma_gradient(self, x):
        return (-self.beta * np.exp(-self.beta * x + self.beta)) / (np.exp(self.beta) - 1)

    @staticmethod
    def _angle(data, prototypes):
        return -1 * (sp.spatial.distance.cdist(data, np.atleast_2d(prototypes), 'cosine') + 1)

    def __call__(self, data, model):
        return np.atleast_2d(self._gamma(self._angle(data, model.prototypes_)))

    def gradient(self, data, model, i_prototype):
        prototype = model.prototypes_[i_prototype, :]
        cossim = self._angle(data, prototype)

        return self._gamma_gradient(cossim) * np.atleast_2d((prototype / (np.linalg.norm(data) * np.linalg.norm(prototype))) - cossim * data / np.linalg.norm(data)**2)

