from . import DistanceBaseClass

import numpy as np
from scipy.spatial.distance import cdist

from ._adaptive_squared_euclidean import (
    # _nan_mahalanobis,
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

    __slots__ = "force_all_finite"

    def __init__(self, force_all_finite=True):
        self.force_all_finite = force_all_finite

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
            The X for which the distance gradient to the prototypes of the model need to be
            computed.
        model : LGMLVQ
            The model instance.

        Returns
        -------
        ndarray
            The adaptive squared euclidean distance for every sample to every prototype stored row-wise.
        """
        prototypes_, omegas_ = model.get_model_params()
        prototypes_labels_ = model.prototypes_labels_
        localization = model.relevance_params["localization"]

        pdists = np.zeros((data.shape[0], model._prototypes_shape[0]))

        if localization == "prototypes":
            for i, (prototype, omega) in enumerate(zip(prototypes_, omegas_)):
                pdists[:, i] = cdist(
                    data,
                    np.atleast_2d(prototype),
                    "mahalanobis",
                    VI=model._compute_lambda(omega),
                ).squeeze()

        if localization == "class":
            for i, omega in enumerate(omegas_):
                # Prototype labels are indices to model.classes_ so all prototypes with 'index'
                # i as label have the same class.
                prototypes = prototypes_[i == prototypes_labels_, :]
                pdists[:, i == prototypes_labels_] = cdist(
                    data,
                    np.atleast_2d(prototypes),
                    "mahalanobis",
                    VI=model._compute_lambda(omega),
                )

        return pdists ** 2

    def gradient(
        self, data: np.ndarray, model: "LGMLVQ", i_prototype: int
    ) -> np.ndarray:
        """ The partial derivative of the adaptive squared euclidean distance function,
        with respect to a specified prototype and the matrix omega:

            .. math::
                \\frac{\\partial d}{\\partial \\vec{w_i}} = -2 \\cdot \\Lambda_j \\cdot (\\vec{x} - \\vec{w_i})

        with :math:`\\Lambda_j` the matrix matched to the prototype. This depends on the
        localization setting of the model.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The X for which the distance gradient to the prototypes of the model need to be
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

        prototype = prototypes[i_prototype, :]
        omega = omegas[model.prototypes_labels_[i_prototype], :, :]

        num_samples, _ = data.shape

        distance_gradient = np.empty(
            (num_samples, prototype.size + omega.size), dtype="float64", order="C"
        )

        # Difference we need for both the prototype and omega part.
        difference = data - prototype

        if self.force_all_finite == "allow-nan":
            difference[np.isnan(difference)] = 0.0

        _prototype_gradient(
            difference, omega, out=distance_gradient[:, : prototype.size]
        )

        distance_gradient_omega_view = distance_gradient[:, prototype.size:].reshape(
            (num_samples, *omega.shape)
        )

        # Omega gradient part
        _omega_gradient(difference, omega, out=distance_gradient_omega_view)

        return distance_gradient
