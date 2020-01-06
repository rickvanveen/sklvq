from scipy.spatial.distance import cdist
import numpy as np

from . import DistanceBaseClass


class CosineDistance(DistanceBaseClass):

    def __init__(self, beta=1):
        self.beta = beta

    @staticmethod
    def _angle(data: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        return -1 * (cdist(data, np.atleast_2d(prototypes), 'cosine') + 1)

    def _gamma(self, angle: np.ndarray) -> np.ndarray:
        return (np.exp(-self.beta * (angle - 1)) - 1) / (np.exp(2 * self.beta) - 1)

    def _gamma_gradient(self, angle: np.ndarray) -> np.ndarray:
        return (-self.beta * np.exp(-self.beta * angle + self.beta)) / (np.exp(self.beta) - 1)

    @staticmethod
    def _angle_gradient(data: np.ndarray, prototype: np.ndarray, angle: np.ndarray) -> np.ndarray:
        return np.atleast_2d((prototype / (np.linalg.norm(data) * np.linalg.norm(prototype))) -
                             (angle * data / np.linalg.norm(data) ** 2))

    def __call__(self, data: np.ndarray, model) -> np.ndarray:
        return np.atleast_2d(self._gamma(self._angle(data, model.prototypes_)))

    def gradient(self, data: np.ndarray, model, i_prototype: int) -> np.ndarray:
        shape = [data.shape[0], *model.prototypes_.shape]
        gradient = np.zeros(shape)

        prototype = model.prototypes_[i_prototype, :]
        angle = self._angle(data, prototype)

        gradient[:, i_prototype, :] = np.atleast_2d(
            self._gamma_gradient(angle) * self._angle_gradient(data, prototype, angle))

        return gradient.reshape(shape[0], shape[1] * shape[2])

