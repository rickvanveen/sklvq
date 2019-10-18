from scipy.spatial.distance import cdist
import numpy as np

from . import DistanceBaseClass


class CosineDistance(DistanceBaseClass):

    def __init__(self, beta=1):
        self.beta = beta

    @staticmethod
    def _angle(data, prototypes):
        return -1 * (cdist(data, np.atleast_2d(prototypes), 'cosine') + 1)

    def _gamma(self, angle):
        return (np.exp(-self.beta * (angle - 1)) - 1) / (np.exp(2 * self.beta) - 1)

    def _gamma_gradient(self, angle):
        return (-self.beta * np.exp(-self.beta * angle + self.beta)) / (np.exp(self.beta) - 1)

    @staticmethod
    def _angle_gradient(data, prototype, angle):
        return np.atleast_2d((prototype / (np.linalg.norm(data) * np.linalg.norm(prototype))) -
                             (angle * data / np.linalg.norm(data) ** 2))

    def __call__(self, data, model):
        return np.atleast_2d(self._gamma(self._angle(data, model.prototypes_)))

    def gradient(self, data, model, i_prototype):
        prototype = model.prototypes_[i_prototype, :]
        angle = self._angle(data, prototype)

        return self._gamma_gradient(angle) * self._angle_gradient(data, prototype, angle)

