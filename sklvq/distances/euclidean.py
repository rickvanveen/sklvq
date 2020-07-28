import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from . import DistanceBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import GLVQ


class Euclidean(DistanceBaseClass):
    """ Euclidean distance function

    See also
    --------
    SquaredEuclidean, AdaptiveEuclidean, AdaptiveSquaredEuclidean, LocalAdaptiveSquaredEuclidean
    """
    def __init__(self, **other_kwargs):
        # Default euclidean
        self.metric_kwargs = {"metric": "euclidean", "squared": False}

        # Could contain other kwargs for sklearn.metrics.pairwise_distances
        if other_kwargs is not None:
            self.metric_kwargs.update(other_kwargs)

        # This might include force_all_finite which if it is set to "allow-nan" should switch the
        # metric used to nan_euclidean else euclidean is fine. It works.
        if "force_all_finite" in self.metric_kwargs:
            if self.metric_kwargs["force_all_finite"] == "allow-nan":
                self.metric_kwargs.update({"metric": "nan_euclidean"})

    def __call__(self, data: np.ndarray, model: "GLVQ") -> np.ndarray:
        """ Computes the Euclidean distance:
            .. math::

                d(\\vec{w}, \\vec{x}) = \\sqrt{(\\vec{x} - \\vec{w})^T (\\vec{x} - \\vec{w})},

            with :math:`\\vec{w}` a prototype and :math:`\\vec{x}` a sample.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GLVQ
            The model instance.

        Returns
        -------
        ndarray with shape (n_samples, n_prototypes)
            Evaluation of the distance between each sample and prototype of the model.
        """
        return pairwise_distances(data, model.prototypes_, **self.metric_kwargs)

    def gradient(self, data: np.ndarray, model: "GLVQ", i_prototype: int) -> np.ndarray:
        """ Implements the derivative of the euclidean distance, with respect to a single
        prototype for the euclidean and nan_euclidean setting.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GLVQ
            The model instance.
        i_prototype : int
            Index of the prototype to compute the gradient for.

        Returns
        -------
        gradient : ndarray with shape (n_samples, n_features)
            The gradient with respect to the prototype and every sample in the data.

        """
        prototypes = model._get_model_params()
        (num_samples, num_features) = data.shape

        # Can also always replace all nans in difference with 0.0, but maybe this is better.
        force_all_finite = self.metric_kwargs.get("force_all_finite", None)

        distance_gradient = np.zeros((num_samples, prototypes.size))

        ip_start = i_prototype * num_features
        ip_end = ip_start + num_features

        difference = np.atleast_2d(data - prototypes[i_prototype, :])

        # Only check for nans if allow-nan is set else it should not happen...
        if force_all_finite == "allow-nan":
            difference[np.isnan(difference)] = 0.0

        denominator = np.sqrt(np.einsum("ij, ij -> i", difference, difference))

        # If data is exactly equal to prototype the denominator will be zero. This happens mostly
        # when nan differences are replaced by zero distance
        denominator[denominator == 0.0] = 1.0 # TODO: huh?

        distance_gradient[:, ip_start:ip_end] = (-1 * difference) / denominator[
            :, np.newaxis
        ]

        return distance_gradient
