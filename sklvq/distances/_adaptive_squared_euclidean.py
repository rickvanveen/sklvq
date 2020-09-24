from . import DistanceBaseClass

import numpy as np
from scipy.spatial.distance import cdist

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GMLVQ


class AdaptiveSquaredEuclidean(DistanceBaseClass):
    """ Adaptive squared Euclidean distance

    See also
    --------
    Euclidean, SquaredEuclidean, LocalAdaptiveSquaredEuclidean

    """

    __slots__ = "force_all_finite"

    def __init__(self, force_all_finite=True):
        self.force_all_finite = force_all_finite

    def __call__(self, data: np.ndarray, model: "GMLVQ") -> np.ndarray:
        """ Implements the adaptive squared euclidean distance:

            .. math::
                d^{\\Lambda}(\\vec{w}, \\vec{x}) = (\\vec{x} - \\vec{w})^{\\top}
                \\Omega^{\\top} \\Omega (\\vec{x} - \\vec{w})

        with :math:`\\Lambda = \\Omega^{\\top} \\Omega`.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The X for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GMLVQ
            The model instance.

        Returns
        -------
        ndarray
            The adaptive squared euclidean distance for every sample to every prototype stored row-wise.
        """
        return (
            cdist(
                data,
                model.prototypes_,
                "mahalanobis",
                VI=model._compute_lambda(model.omega_),
            )
            ** 2
        )

    def gradient(
        self, data: np.ndarray, model: "GMLVQ", i_prototype: int
    ) -> np.ndarray:
        """ The partial derivative of the adaptive squared euclidean distance function,
        with respect to a specified prototype and the matrix omega:

            .. math::
                \\frac{\\partial d}{\\partial \\vec{w_i}} = -2 \\cdot \\Lambda \\cdot (\\vec{x} -
                \\vec{w_i})

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The X for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GMLVQ
            The model instance.
        i_prototype : int
            An integer index value of the relevant prototype

        Returns
        -------
        ndarray
            The gradient for every feature/dimension. Returned in one 1D vector. The non-relevant prototype's
            gradient is set to 0, but is still included in the output.
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


# def _nan_mahalanobis(sample, prototype, VI=None):
#     difference = sample - prototype
#     difference[np.isnan(difference)] = 0.0
#     # Equal to difference.dot(VI).dot(difference)
#     return np.einsum("i, ij, i ->", difference, VI, difference)


def _prototype_gradient(
    difference: np.ndarray, omega: np.ndarray, out=None
) -> np.ndarray:
    return np.einsum("ji,ik ->jk", -2.0 * difference, np.dot(omega.T, omega), out=out)
    # Other (slower) variants:
    #   - np.dot(-2.0 * difference.dot(omega.T.dot(omega))
    #   - np.einsum("ij, kj, kl -> il", -2.0 * difference, omega, omega)


def _omega_gradient(difference: np.ndarray, omega: np.ndarray, out=None) -> np.ndarray:
    return np.einsum(
        "ij,jk->jik", np.dot(omega, difference.T), (2.0 * difference), out=out
    )
    # Other (slower) variants:
    #   - np.einsum("ij, kj, kl -> kil", omega, difference, (2.0 * difference)) this is slower
