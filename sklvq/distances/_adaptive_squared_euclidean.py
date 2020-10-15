from . import DistanceBaseClass

import numpy as np
from scipy.spatial.distance import cdist

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GMLVQ


class AdaptiveSquaredEuclidean(DistanceBaseClass):
    """ Adaptive squared Euclidean distance

    Class that holds the adaptive squared Euclidean distance function and its gradient as
    described in [1]_ and [2]_.

    Parameters
    ----------
    force_all_finite  : {True, False, "allow-nan"}
        Parameter to indicate that NaNLVQ distance variant should be used. If true no nans are
        allowed. If False or "allow-nan", nans are allowed.

    See also
    --------
    Euclidean, SquaredEuclidean, LocalAdaptiveSquaredEuclidean

    Notes
    -----
    Compatible with the :class:`.GMLVQ` algorithm (only).

    References
    ----------
    .. [1] Schneider, P. (2010). Advanced methods for prototype-based classification. Groningen.
    .. [2] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive Relevance Matrices in Learning
           Vector Quantization. Neural Computation, 21(12), 3532–3561.

    """

    __slots__ = "force_all_finite"

    def __init__(self, force_all_finite=True):
        self.force_all_finite = force_all_finite

    def __call__(self, data: np.ndarray, model: "GMLVQ") -> np.ndarray:
        """ Computes the adaptive squared Euclidean distance:

            .. math::
                d^{\\Lambda}(\\vec{w}, \\vec{x}) = (\\vec{x} - \\vec{w})^{\\top}
                \\Lambda (\\vec{x} - \\vec{w})

            with the relevance matrix :math:`\\Lambda = \\Omega^{\\top} \\Omega`, the  prototype
            :math:`\\vec{w}`, and sample :math:`\\vec{x}`.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GMLVQ
            The model instance, containing the prototypes and relevance matrix.

        Returns
        -------
        ndarray with shape (n_samples, n_prototypes)
           Evaluation of the distance between each sample in the data and prototype of the model.
        """

        distance_function = "mahalanobis"
        if self.force_all_finite == "allow-nan" or False:
            distance_function = _nan_mahalanobis

        return (
            cdist(
                data,
                model.prototypes_,
                distance_function,
                VI=model._compute_lambda(model.omega_),
            )
            ** 2
        )

    def gradient(
        self, data: np.ndarray, model: "GMLVQ", i_prototype: int
    ) -> np.ndarray:
        """ Computes the gradient of the adaptive squared euclidean distance function,
        with respect to a single prototype:

            .. math::
                \\frac{\\partial d}{\\partial \\vec{w_i}} = -2 \\Lambda (\\vec{x} -
                \\vec{w_i}),

        and the omega matrix (per element):

            .. math::
                \\frac{\\partial d}{\\partial \\Omega_{lm}} =  2 \\sum_i (x^i - w^i)
                \\Omega_{li} (x^m - w^m)

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GMLVQ
            The model instance, containing the prototypes and relevance matrix.
        i_prototype : int
            An integer index value of the relevant prototype

        Returns
        -------
        ndarray with shape (n_samples, n_features + n_omega_elements)
            The gradient of the prototype and omega matrix with respect to each data sample.
        """

        (prototypes, omega) = model.get_model_params()

        prototype = prototypes[i_prototype, :]

        (num_samples, _) = data.shape

        distance_gradient = np.empty(
            (num_samples, prototype.size + omega.size), dtype="float64", order="C"
        )

        difference = data - prototype

        if self.force_all_finite == "allow-nan":
            difference[np.isnan(difference)] = 0.0

        # Prototype gradient directly computed in the distance_gradient
        _prototype_gradient(
            difference, omega, out=distance_gradient[:, : prototype.size]
        )

        # Omega view created in to hold the shape the _omega_gradient functions needs to output in.
        distance_gradient_omega_view = distance_gradient[:, prototype.size :].reshape(
            (num_samples, *omega.shape)
        )
        # Omega gradient indirectly computed in the distance_gradient via the view of the memory
        # in a different shape.
        _omega_gradient(difference, omega, out=distance_gradient_omega_view)

        return distance_gradient


def _nan_mahalanobis(sample, prototype, VI=None):
    # The NaNLVQ variant of the mahalanobis distance
    difference = sample - prototype
    difference[np.isnan(difference)] = 0.0
    # Equal to difference.dot(VI).dot(difference)
    return np.einsum("i, ij, i ->", difference, VI, difference)


def _prototype_gradient(
    difference: np.ndarray, omega: np.ndarray, out=None
) -> np.ndarray:
    # The gradient with respect to a prototype. Equivalent to np.dot(-2.0 * difference.dot(
    # omega.T.dot(omega)).
    return np.einsum("ji,ik ->jk", -2.0 * difference, np.dot(omega.T, omega), out=out)


def _omega_gradient(difference: np.ndarray, omega: np.ndarray, out=None) -> np.ndarray:
    # The gradient with respect to the omega matrix.
    return np.einsum(
        "ij,jk->jik", np.dot(omega, difference.T), (2.0 * difference), out=out
    )
