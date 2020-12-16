from . import DistanceBaseClass

import numpy as np
from scipy.spatial.distance import cdist

from ._adaptive_squared_euclidean import (
    _nan_mahalanobis,
    _prototype_gradient,
    _omega_gradient,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import LGMLVQ


class LocalAdaptiveSquaredEuclidean(DistanceBaseClass):
    """Local adaptive squared Euclidean distance

    Class that holds the localized adaptive squared Euclidean distance function and its gradient as
    described in `[1]`_ and `[2]`_.

    Parameters
    ----------
    force_all_finite  : {True, False, "allow-nan"}
        Parameter to indicate that NaNLVQ distance variant should be used. If true no nans are
        allowed. If False or "allow-nan" nans are allowed.

    See also
    --------
    Euclidean, SquaredEuclidean, AdaptiveSquaredEuclidean

    Notes
    -----
    Compatible with the :class:`.LGMLVQ` algorithm (only).

    References
    ----------
    _`[1]` Schneider, P. (2010). Advanced methods for prototype-based classification. Groningen.

    _`[2]` Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive Relevance Matrices in Learning
    Vector Quantization. Neural Computation, 21(12), 3532–3561.
    """

    __slots__ = ()

    def __call__(self, data: np.ndarray, model: "LGMLVQ") -> np.ndarray:
        r"""Computes the local variant of the adaptive squared Euclidean distance:

            .. math::
                d^{\Lambda}(\mathbf{w}, \mathbf{x}) = (\mathbf{x} - \mathbf{w})^{\top}
                \Omega_j^{\top} \Omega_j (\mathbf{x} - \mathbf{w})

        with :math:`\Omega_j` depending on the localization setting of the model and
        :math:`\Lambda_j = \Omega_j^{\top} \Omega_j`. The localization can be either per
        prototype or per class, see the documentation of :class:`.LGMLVQ`.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : LGMLVQ
            A LGMLVQ model instance, containing the prototypes and relevance matrices.

        Returns
        -------
        ndarray with shape (n_samples, n_prototypes)
            Evaluation of the distance between each sample in the data and prototype of the model.
        """
        prototypes_, omegas_ = model.get_model_params()
        prototypes_labels_ = model.prototypes_labels_

        distance_function = "mahalanobis"
        kwarg_str = "VI"

        if model.force_all_finite == "allow-nan":
            distance_function = _nan_mahalanobis
            # RM because VI is filtered out of the  kwargs by cdist...
            kwarg_str = "RM"

        cdists = np.zeros((data.shape[0], model._prototypes_shape[0]))

        if model.relevance_localization == "prototypes":
            for i, (prototype, omega) in enumerate(zip(prototypes_, omegas_)):
                cdists[:, i] = cdist(
                    data,
                    np.atleast_2d(prototype),
                    distance_function,
                    **{kwarg_str: model._compute_lambda(omega)},
                ).squeeze()

        if model.relevance_localization == "class":
            for i, omega in enumerate(omegas_):
                # Prototype labels are indices to model.classes_ so all prototypes with 'index'
                # i as label have the same class.
                prototypes = prototypes_[i == prototypes_labels_, :]
                cdists[:, i == prototypes_labels_] = cdist(
                    data,
                    np.atleast_2d(prototypes),
                    distance_function,
                    **{kwarg_str: model._compute_lambda(omega)},
                )

        return cdists ** 2

    def gradient(
        self, data: np.ndarray, model: "LGMLVQ", i_prototype: int
    ) -> np.ndarray:
        r"""Computes the gradient of the localized adaptive squared euclidean distance function
        with respect to a specified prototype:

            .. math::
                \frac{\partial d}{\partial \mathbf{w}_i} = -2 \Lambda_j (\mathbf{x} - \mathbf{w}_i)

        and implicitly coupled omega matrix (per element):

            .. math::
                \frac{\partial d}{\partial \Omega_{lm}} =  2 \sum_i (x^i - w^i)
                \Omega_{li} (x^m - w^m)

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The X for which the distance gradient to the prototypes of the model need to be
            computed.
        model : LGMLVQ
            The LGMLVQ model instance, containing the prototypes and relevance matrices.
        i_prototype : int
            An integer index value of the relevant prototype

        Returns
        -------
        ndarray with shape (n_samples, n_features + n_omega_elements)
            The gradient of the prototype and omega matrix with respect to each data sample.
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

        if model.force_all_finite == "allow-nan":
            difference[np.isnan(difference)] = 0.0

        _prototype_gradient(
            difference, omega, out=distance_gradient[:, : prototype.size]
        )

        distance_gradient_omega_view = distance_gradient[:, prototype.size :].reshape(
            (num_samples, *omega.shape)
        )

        # Omega gradient part
        _omega_gradient(difference, omega, out=distance_gradient_omega_view)

        return distance_gradient
