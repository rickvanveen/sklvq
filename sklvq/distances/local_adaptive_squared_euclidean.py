from . import DistanceBaseClass

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from .adaptive_squared_euclidean import (
    _nan_mahalanobis,
    _prototype_gradient,
    _omega_gradient,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import LGMLVQ


class LocalAdaptiveSquaredEuclidean(DistanceBaseClass):
    """ Local adaptive squared Euclidean distance

    See also
    --------
    Euclidean, SquaredEuclidean, AdaptiveSquaredEuclidean

    """

    def __init__(self, **kwargs):
        self.metric_kwargs = {
            "metric": "mahalanobis",
        }

        if kwargs is not None:
            self.metric_kwargs.update(kwargs)

        if "force_all_finite" in self.metric_kwargs:
            if self.metric_kwargs["force_all_finite"] == "allow-nan":
                self.metric_kwargs.update({"metric": _nan_mahalanobis})

    def __call__(self, data: np.ndarray, model: "LGMLVQ") -> np.ndarray:
        """ Implements a weighted variant of the squared euclidean distance:

            .. math::
                d^{\\Lambda}(\\vec{w}, \\vec{x}) = (\\vec{x} - \\vec{w})^{\\top}
                \\Omega_j^{\\top} \\Omega_j (\\vec{x} - \\vec{w})

        with :math:`\\Omega_j` depending on the localization setting of the model and
        :math:`\\Lambda_j = \\Omega_j^{\\top} \\Omega_j`

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : LGMLVQ
            The model instance.

        Returns
        -------
        ndarray
            The adaptive squared euclidean distance for every sample to every prototype stored row-wise.
        """

        pdists = np.zeros((data.shape[0], model.prototypes_.shape[0]))

        prototypes_, omega_ = model.get_model_params()
        prototypes_labels_ = model.prototypes_labels_

        if model.localization == "prototype":
            for i, (prototype, omega) in enumerate(zip(prototypes_, omega_)):
                self.metric_kwargs.update(dict(VI=omega.T.dot(omega)))

                pdists[:, i] = pairwise_distances(
                    data, np.atleast_2d(prototype), **self.metric_kwargs
                ).squeeze()

        if model.localization == "class":
            for i, omega in enumerate(model.omega_):
                # Prototype labels are indices to model.classes_ so all prototypes with 'index'
                # i as label have the same class.
                prototypes = prototypes_[i == prototypes_labels_, :]

                self.metric_kwargs.update(dict(VI=omega.T.dot(omega)))
                pdists[:, i == prototypes_labels_] = pairwise_distances(
                    data, np.atleast_2d(prototypes), **self.metric_kwargs
                )

        return pdists ** 2

    def gradient(
        self, data: np.ndarray, model: "LGMLVQ", i_prototype: int
    ) -> np.ndarray:
        """ The partial derivative of the adaptive squared euclidean distance function, with respect
        to a specified prototype and the matrix omega.

            .. math::
                \\frac{\\partial d}{\\partial \\vec{w_i}} = -2 \\cdot \\Lambda_j \cdot (\\vec{x} -
                \\vec{w_i})

        with :math:`\\Lambda_j` the matrix matched to the prototype. This depends on the
        localization setting of the model.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : LGMLVQ
            The model instance.
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

        force_all_finite = self.metric_kwargs.get("force_all_finite", None)

        distance_gradient = np.zeros((num_samples, prototypes.size + omegas.size))

        ip_start = i_prototype * num_features
        ip_end = ip_start + num_features

        # prototype labels are indices of model.classes_. Every prototype is (implicitly) linked
        # to omega by either  class or per prototype. There are no labels for omega.
        i_omega = model.prototypes_labels_[i_prototype]
        omega = omegas[i_omega, :, :]

        # Difference we need for both the prototype and omega part.
        difference = data - prototypes[i_prototype, :]
        if force_all_finite == "allow-nan":
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
