from . import DistanceBaseClass

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass


class AdaptiveSquaredEuclidean(DistanceBaseClass):
    """ Adaptive squared Euclidean function

    See also
    --------
    """

    def __init__(self, **other_kwargs):
        self.metric_kwargs = {
            "metric": "mahalanobis",
        }

        if other_kwargs is not None:
            self.metric_kwargs.update(other_kwargs)

        if "force_all_finite" in self.metric_kwargs:
            if self.metric_kwargs["force_all_finite"] == "allow-nan":
                self.metric_kwargs.update({"metric": _nan_mahalanobis})

    def __call__(self, data: np.ndarray, model: "LVQBaseClass") -> np.ndarray:
        """ Implements a weighted variant of the squared euclidean distance:
            .. math::
                d^{\\Lambda}(w, x) = (x - w)^T \\Lambda (x - w)

        Parameters
        ----------
        data : numpy.ndarray
            A matrix containing the samples on the rows.
        model : LVQBaseClass, GMLVQ

        Returns
        -------
        numpy.ndarray
            The adaptive squared euclidean distance for every sample to every prototype stored row-wise.
        """
        (prototypes, omega) = model.get_model_params()

        self.metric_kwargs.update(dict(VI=np.dot(omega.T, omega)))

        pdists = pairwise_distances(data, prototypes, **self.metric_kwargs)

        if self.metric_kwargs["metric"] == "mahalanobis":
            return pdists ** 2

        return pdists

    def gradient(
        self, data: np.ndarray, model: "LVQBaseClass", i_prototype: int
    ) -> np.ndarray:
        """ The partial derivative of the adaptive squared euclidean distance function, with respect
        to a specified prototype and the matrix omega.

        Parameters
        ----------
        data : ndarray
            A matrix containing the samples on the rows.
        model : LVQBaseClass
            In principle any LVQClassifier that calls it's relevance matrix omega.
            Specifically here, GMLVQ.
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
        distance_gradient[:, ip_start:ip_end] = np.einsum(
            "ji,ik ->jk", -2.0 * difference, np.dot(omega.T, omega)
        )

        # Omega gradient
        scaled_omega = np.dot(omega, difference.T)
        distance_gradient[:, io_start:] = (
            np.einsum("ij,jk->jik", scaled_omega, (2.0 * difference))
        ).reshape(num_samples, omega.size)

        return distance_gradient


def _nan_mahalanobis(sample, prototype, VI=None):
    difference = sample - prototype
    difference[np.isnan(difference)] = 0.0
    # Equal to difference.dot(VI).dot(difference)
    return np.einsum("i, ij, i ->", difference, VI, difference)
