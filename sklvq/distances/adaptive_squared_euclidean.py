from . import DistanceBaseClass

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from typing import TYPE_CHECKING
from typing import Dict

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class AdaptiveSquaredEuclidean(DistanceBaseClass):
    def __init__(self, other_kwargs: Dict = None):
        self.metric_kwargs = {
            "metric": "mahalanobis",
        }

        if other_kwargs is not None:
            self.metric_kwargs.update(other_kwargs)

    def __call__(self, data: np.ndarray, model: "LVQClassifier") -> np.ndarray:
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
        # model.get_prototypes()
        # model.get_omega()
        # model.get_lambda()

        self.metric_kwargs.update({"VI": np.dot(model.omega_.T, model.omega_)})

        return pairwise_distances(data, model.prototypes_, **self.metric_kwargs) ** 2

    def gradient(
        self, data: np.ndarray, model: "LVQClassifier", i_prototype: int
    ) -> np.ndarray:
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
            _prototype_gradient(data, model.prototypes_[i_prototype, :], model.omega_)
        )
        omega_gradient = np.atleast_2d(
            _omega_gradient(data, model.prototypes_[i_prototype, :], model.omega_)
        )

        # TODO In variables form and reshaped by model.to_params(variables)
        return np.hstack(
            (prototype_gradient.reshape(shape[0], shape[1] * shape[2]), omega_gradient)
        )


def _prototype_gradient(
    data: np.ndarray, prototype: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    # difference = (-2 * (data - prototype)).T
    # relevance = omega.T.dot(omega)
    # return np.matmul(relevance, difference).T
    return np.einsum("ji,ik ->jk", -2 * (data - prototype), np.dot(omega.T, omega))
    # return np.atleast_2d(
    #     np.einsum("ji, ik, kj -> i", omega, omega, -2 * (data - prototype))
    # )


def _omega_gradient(
    data: np.ndarray, prototype: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    difference = data - prototype
    scaled_omega = np.dot(omega, difference.T)
    return np.einsum("ij,jk->jik", scaled_omega, (2 * difference)).reshape(
        data.shape[0], omega.shape[0] * omega.shape[1]
    )
