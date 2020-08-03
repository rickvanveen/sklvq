from . import DistanceBaseClass

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GMLVQ


class AdaptiveSquaredEuclidean(DistanceBaseClass):
    """ Adaptive squared Euclidean distance

    See also
    --------
    Euclidean, SquaredEuclidean, LocalAdaptiveSquaredEuclidean

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

    def __call__(self, data: np.ndarray, model: "GMLVQ") -> np.ndarray:
        """ Implements the adaptive squared euclidean distance:

            .. math::
                d^{\\Lambda}(\\vec{w}, \\vec{x}) = (\\vec{x} - \\vec{w})^{\\top}
                \\Omega^{\\top} \\Omega (\\vec{x} - \\vec{w})

        with :math:`\\Lambda = \\Omega^{\\top} \\Omega`.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GMLVQ
            The model instance.

        Returns
        -------
        ndarray
            The adaptive squared euclidean distance for every sample to every prototype stored row-wise.
        """
        (prototypes, omega) = model.get_model_params()

        self.metric_kwargs.update(dict(VI=np.dot(omega.T, omega)))

        pdists = pairwise_distances(data, prototypes, **self.metric_kwargs)

        if self.metric_kwargs["metric"] == "mahalanobis":
            return pdists ** 2

        return pdists

    def gradient(
        self, data: np.ndarray, model: "GMLVQ", i_prototype: int
    ) -> np.ndarray:
        r""" The partial derivative of the adaptive squared euclidean distance function,
        with respect to a specified prototype and the matrix omega:

            .. math::
                \\frac{\\partial d}{\\partial \\vec{w_i}} = -2 \\cdot \\Lambda \cdot (\\vec{x} -
                \\vec{w_i})

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
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
        (num_samples, num_features) = data.shape

        force_all_finite = self.metric_kwargs.get("force_all_finite", None)

        distance_gradient = np.zeros((num_samples, prototypes.size + omega.size))

        # Start and end indices prototype
        ip_start = i_prototype * num_features
        ip_end = ip_start + num_features

        # Start index omega
        io_start = prototypes.size

        # If nans are allowed remove them from the difference and replace it with 0.0
        difference = data - prototypes[i_prototype, :]
        if force_all_finite == "allow-nan":
            difference[np.isnan(difference)] = 0.0

        # Prototype gradient
        distance_gradient[:, ip_start:ip_end] = _prototype_gradient(difference, omega)

        # Omega gradient
        distance_gradient[:, io_start:] = _omega_gradient(difference, omega).reshape(
            num_samples, omega.size
        )

        return distance_gradient


def _nan_mahalanobis(sample, prototype, VI=None):
    difference = sample - prototype
    difference[np.isnan(difference)] = 0.0
    # Equal to difference.dot(VI).dot(difference)
    return np.einsum("i, ij, i ->", difference, VI, difference)


def _prototype_gradient(difference: np.ndarray, omega: np.ndarray) -> np.ndarray:
    # np.dot(-2.0 * difference.dot(omega.T.dot(omega))
    return np.einsum("ji,ik ->jk", -2.0 * difference, np.dot(omega.T, omega))
    # return np.einsum("ij, kj, kl -> il", -2.0 * difference, omega, omega) this is slower


def _omega_gradient(difference: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk->jik", np.dot(omega, difference.T), (2.0 * difference))
    # return np.einsum("ij, kj, kl -> kil", omega, difference, (2.0 * difference)) this is slower
