from . import DistanceBaseClass

import numpy as np
import scipy as sp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class AdaptiveSquaredEuclidean(DistanceBaseClass):

    def __call__(self, data: np.ndarray, model: 'LVQClassifier') -> np.ndarray:
        """ Implements a weighted variant of the squared euclidean distance:
            .. math::
                d^{\\Lambda}(w, x) = (x - w)^T \\Lambda (x - w)

        .. note::
            Uses scipy.spatial.distance.cdist, see scipy documentation for more detail.

        Parameters
        ----------
        data : ndarray
            A matrix containing the samples on the rows.
        model : LVQClassifier
            In principle any LVQClassifier that calls it's relevance matrix omega.
            Specifically here, GMLVQClassifier.

        Returns
        -------
        ndarray
            The adaptive squared euclidean distance for every sample to every prototype stored row-wise.
        """
        return sp.spatial.distance.cdist(data, model.prototypes_, 'mahalanobis',
                                         VI=model.omega_.T.dot(model.omega_)) ** 2

    def gradient(self, data: np.ndarray, model: 'LVQClassifier', i_prototype: int) -> np.ndarray:
        """ The partial derivative of the adaptive squared euclidean distance function, with respect
        to a specified prototype and the matrix omega.

        Parameters
        ----------
        data : ndarray
            A matrix containing the samples on the rows.
        model : LVQClassifier
            In principle any LVQClassifier that calls it's relevance matrix omega.
            Specifically here, GMLVQClassifier.
        i_prototype : int
            An integer index value of the relevant prototype

        Returns
        -------
        ndarray
            The gradient for every feature/dimension. Returned in one 1D vector. The non-relevant prototype's
            gradient is set to 0, but is still included in the output.
        """
        shape = [data.shape[0], *model.prototypes_.shape]
        prototype_gradient = np.zeros(shape)

        prototype_gradient[:, i_prototype, :] = np.atleast_2d(
            _prototype_gradient(data,
                                model.prototypes_[i_prototype, :],
                                model.omega_)
        )
        omega_gradient = np.atleast_2d(
            _omega_gradient(data,
                            model.prototypes_[i_prototype, :],
                            model.omega_)
        )

        return np.hstack((prototype_gradient.reshape(shape[0], shape[1] * shape[2]), omega_gradient))


def _prototype_gradient(data: np.ndarray, prototype: np.ndarray, omega: np.ndarray) -> np.ndarray:
    difference = (-2 * (data - prototype)).T
    relevance = omega.T.dot(omega)
    return np.matmul(relevance, difference).T


def _omega_gradient(data: np.ndarray, prototype: np.ndarray, omega: np.ndarray) -> np.ndarray:
    difference = data - prototype
    scaled_omega = omega.dot(difference.T)
    scaled_diff = 2 * difference
    return np.einsum('ij,jk->jik', scaled_omega, scaled_diff).reshape(data.shape[0],
                                                                      omega.shape[0] * omega.shape[1])
