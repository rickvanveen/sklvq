from . import DistanceBaseClass

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklvq.distances.adaptive_squared_euclidean import _nan_mahalanobis

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LGMLVQ


class LocalAdaptiveSquaredEuclidean(DistanceBaseClass):
    def __init__(self, **other_kwargs):
        self.metric_kwargs = {
            "metric": "mahalanobis",
        }

        if other_kwargs is not None:
            self.metric_kwargs.update(other_kwargs)

        if "force_all_finite" in self.metric_kwargs:
            if self.metric_kwargs["force_all_finite"] == "allow-nan":
                self.metric_kwargs.update({"metric": _nan_mahalanobis})

    def __call__(self, data: np.ndarray, model: "LGMLVQ") -> np.ndarray:
        """ Implements a weighted variant of the squared euclidean distance:
            .. math::
                d^{\\Lambda}(w, x) = (x - w)^T \\Lambda (x - w)

        Parameters
        ----------
        data : ndarray
            A matrix containing the samples on the rows.
        model : LGMLVQ
            In principle any LVQBaseClass with relevant properties (omega, localization),
            but specifically here: LGMLVQ.

        Returns
        -------
        ndarray
            The adaptive squared euclidean distance for every sample to every prototype stored row-wise.
        """

        pdists = np.zeros((data.shape[0], model.prototypes_.shape[0]))

        # distance depends on prototype and localizaton setting....
        # if model.localization == "prototype":
        #     for i, (prototype, omega) in enumerate(
        #         zip(model.prototypes_, model.omega_)
        #     ):
        #         pdists[:, i] = pairwise_distances(
        #             np.atleast_2d(data),
        #             np.atleast_2d(prototype),
        #             metric="mahalanobis",
        #             VI=omega.T.dot(omega),
        #         ).ravel()
        # if model.localization == "class":
        #     for i, omega in enumerate(model.omega_):
        #         pdists[:, i == model.prototypes_labels_] = pairwise_distances(
        #             np.atleast_2d(data),
        #             np.atleast_2d(model.prototypes_[i == model.prototypes_labels_, :]),
        #             metric="mahalanobis",
        #             VI=omega.T.dot(omega),
        #         )

        if model.localization == "prototype":
            for i, (prototype, omega) in enumerate(
                zip(model.prototypes_, model.omega_)
            ):
                self.metric_kwargs.update(dict(VI=omega.T.dot(omega)))

                pdists[:, i] = pairwise_distances(
                    data, np.atleast_2d(prototype), **self.metric_kwargs
                ).squeeze()

        if model.localization == "class":
            for i, omega in enumerate(model.omega_):
                # Prototype labels are indices to model.classes_ so all prototypes with 'index'
                # i as label have the same class.
                prototypes = model.prototypes_[i == model.prototypes_labels_, :]

                self.metric_kwargs.update(dict(VI=omega.T.dot(omega)))
                pdists[:, i == model.prototypes_labels_] = pairwise_distances(
                    data,
                    np.atleast_2d(prototypes),
                    **self.metric_kwargs
                )

        return pdists ** 2

    def gradient(
        self, data: np.ndarray, model: "LGMLVQ", i_prototype: int
    ) -> np.ndarray:
        """ The partial derivative of the adaptive squared euclidean distance function, with respect
        to a specified prototype and the matrix omega.

        Parameters
        ----------
        data : ndarray
            A matrix containing the samples on the rows.
        model : LGMLVQ
            In principle any LVQBaseClass with relevant properties (omega, localization),
            but specifically here: LGMLVQ.
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

        # prototype labels are indices of model.classes_. Every prototype is (implicitly) linked
        # to omega by either  class or per prototype. There are no labels for omega.
        i_omega = model.prototypes_labels_[i_prototype]
        omega = omegas[i_omega, :, :]

        # Difference we need for both the prototype and omega part.
        difference = data - prototypes[i_prototype, :]
        difference[np.isnan(difference)] = 0.0

        # Prototype gradient part
        distance_gradient[:, ip_start:ip_end] = _prototype_gradient(difference, omega)

        io_start = prototypes.size + (i_omega * omega.size)
        io_end = io_start + omega.size

        # Omega gradient part
        distance_gradient[:, io_start:io_end] = _omega_gradient(
            difference, omega
        ).reshape(num_samples, omega.size)

        return distance_gradient


def _prototype_gradient(difference: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.einsum("ji,ik ->jk", -2.0 * difference, np.dot(omega.T, omega))


def _omega_gradient(difference: np.ndarray, omega: np.ndarray) -> np.ndarray:
    scaled_omega = omega.dot(difference.T)
    return np.einsum("ij,jk->jik", scaled_omega, (2.0 * difference))
