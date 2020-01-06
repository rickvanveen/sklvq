from . import DistanceBaseClass

import numpy as np
import scipy as sp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class AdaptiveSquaredEuclidean(DistanceBaseClass):

    def __call__(self, data: np.ndarray, model) -> np.ndarray:
        """ Implements a weighted variant of the squared euclidean distance.

                Note uses scipy.spatial.distance.cdist see scipy documentation.

                Note that any custom function should still accept and return the same as this function.

                Parameters
                ----------
                data       : ndarray, shape = [n_obervations, n_features]
                             Inputs are converted to float type.
                prototypes : ndarray, shape = [n_prototypes, n_features]
                             Inputs are converted to float type.
                omega      : ndarray, shape = [n_features, n_features]

                Returns
                -------
                distances : ndarray, shape = [n_observations, n_prototypes]
                    The dist(u=XA[i], v=XB[j]) is computed and stored in the
                    ij-th entry.
            """
        return sp.spatial.distance.cdist(data, model.prototypes_, 'mahalanobis',
                                         VI=model.omega_.T.dot(model.omega_)) ** 2

    # TODO local matrices
    def gradient(self, data: np.ndarray, model, i_prototype: int) -> np.ndarray:
        shape = [data.shape[0], *model.prototypes_.shape]
        prototype_gradient = np.zeros(shape)

        prototype_gradient[:, i_prototype, :] = np.atleast_2d(
            _prototype_gradient(data,
                                model.prototypes_[i_prototype, :],
                                model.omega_))
        omega_gradient = np.atleast_2d(
            _omega_gradient(data,
                            model.prototypes_[i_prototype, :],
                            model.omega_))

        return np.hstack((prototype_gradient.reshape(shape[0], shape[1] * shape[2]), omega_gradient))


def _prototype_gradient(data: np.ndarray, prototype: np.ndarray, omega: np.ndarray) -> np.ndarray:
    direction = (-2 * (data - prototype)).T
    relevance = omega.T.dot(omega)
    return np.matmul(relevance, direction).T


def _omega_gradient(data: np.ndarray, prototype: np.ndarray, omega: np.ndarray) -> np.ndarray:
    difference = data - prototype
    scaled_omega = omega.dot(difference.T)
    scaled_diff = 2 * difference
    return np.einsum('ij,jk->jik', scaled_omega, scaled_diff).reshape(data.shape[0],
                                                                      omega.shape[0] * omega.shape[1])
