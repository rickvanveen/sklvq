import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from . import DistanceBaseClass

from typing import TYPE_CHECKING
from typing import Dict

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass


class Euclidean(DistanceBaseClass):
    """ Euclidean distance function

    See also
    --------
    SquaredEuclidean, AdaptiveEuclidean, AdaptiveSquaredEuclidean
    """

    def __init__(self, other_kwargs: Dict = None):
        self.metric_kwargs = {
            "metric": "euclidean",
            "squared": False
        }

        if other_kwargs is not None:
            self.metric_kwargs.update(other_kwargs)

    def __call__(self, data: np.ndarray, model: "LVQBaseClass") -> np.ndarray:
        """
        Computes the Euclidean distance:
            .. math::

                d(\\vec{w}, \\vec{x}) = \\sqrt{(\\vec{x} - \\vec{w})^T (\\vec{x} - \\vec{w})},

            with :math:`\\vec{w}` a prototype and :math:`\\vec{x}` a sample.

        Parameters
        ----------
        data : numpy.ndarray with shape (n_samples, n_features)
        model : LVQBaseClass
            Can be any LVQClassifier but only prototypes will be used to compute the distance

        Returns
        -------
        distances : numpy.ndarray with shape (n_observations, n_prototypes)
            The dist(u=XA[i], v=XB[j]) is computed and stored in the
            ij-th entry.
        """
        return pairwise_distances(
            data,
            model.prototypes_,
            **self.metric_kwargs
        )

    def gradient(self, data: np.ndarray, model, i_prototype: int) -> np.ndarray:
        """ Implements the derivative of the euclidean distance, with respect to a single prototype

        Parameters
        ----------
        data : numpy.ndarray with shape (n_samples, n_features)
        model : LVQBaseClass
            Only prototypes need to be available in the LVQClassifier
        i_prototype : int
            Index of the prototype to compute the gradient for

        Returns
        -------
        gradient : ndarray with shape (n_samples, n_features)
            The gradient with respect to the prototype and every sample in data.

        """
        prototypes = model.get_model_params()
        (num_samples, num_features) = data.shape

        distance_gradient = np.zeros((num_samples, prototypes.size))

        ip_start = i_prototype * num_features
        ip_end = ip_start + num_features

        difference = data - prototypes[i_prototype, :]

        # TODO: broken....
        distance_gradient[:, ip_start:ip_end] = (-1 * difference) / np.sqrt(np.dot(difference.T, difference))

        return distance_gradient
