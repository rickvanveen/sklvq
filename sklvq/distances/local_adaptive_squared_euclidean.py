from . import DistanceBaseClass

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from typing import TYPE_CHECKING
from typing import Dict

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class LocalAdaptiveSquaredEuclidean(DistanceBaseClass):
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

        Parameters
        ----------
        data : ndarray
            A matrix containing the samples on the rows.
        model : LVQClassifier
            In principle any LVQClassifier that calls it's relevance matrix omega.
            Specifically here, LGMLVQClassifier.

        Returns
        -------
        ndarray
            The adaptive squared euclidean distance for every sample to every prototype stored row-wise.
        """

        pdists = np.zeros((data.shape[0], model.prototypes_.shape[0]))

        # distance depends on prototype and localizaton setting....
        if model.localization == "p":
            for i, (prototype, omega) in enumerate(
                zip(model.prototypes_, model.omega_)
            ):
                pdists[:, i] = pairwise_distances(
                    np.atleast_2d(data),
                    np.atleast_2d(prototype),
                    metric="mahalanobis",
                    VI=omega.T.dot(omega),
                ).ravel()

        if model.localization == "c":
            for i, omega in enumerate(model.omega_):
                pdists[:, i == model.prototypes_labels_] = pairwise_distances(
                    np.atleast_2d(data),
                    np.atleast_2d(model.prototypes_[i == model.prototypes_labels_, :]),
                    metric="mahalanobis",
                    VI=omega.T.dot(omega),
                )

        return pdists

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
        (prototypes, omegas) = model.get_model_params()
        (num_samples, num_features) = data.shape

        distance_gradient = np.zeros((num_samples, prototypes.size + omegas.size))

        ip_start = i_prototype * num_features
        ip_end = ip_start + num_features

        # prototype labels are indices of model.classes_
        i_omega = model.prototypes_labels_[i_prototype]
        omega = omegas[i_omega, :, :]

        distance_gradient[:, ip_start:ip_end] = _prototype_gradient(
            data, prototypes[i_prototype, :], omega
        )

        io_start = prototypes.size + (i_omega * omega.size)
        io_end = io_start + omega.size

        distance_gradient[:, io_start:io_end] = _omega_gradient(
            data, prototypes[i_prototype, :], omega
        ).reshape(num_samples, omega.size)

        return distance_gradient


def _prototype_gradient(
    data: np.ndarray, prototype: np.ndarray, omega: np.ndarray
) -> np.ndarray:

    return np.einsum("ji,ik ->jk", -2 * (data - prototype), np.dot(omega.T, omega))


def _omega_gradient(
    data: np.ndarray, prototype: np.ndarray, omega: np.ndarray
) -> np.ndarray:

    difference = data - prototype
    scaled_omega = np.dot(omega, difference.T)

    return np.einsum("ij,jk->jik", scaled_omega, (2 * difference))
