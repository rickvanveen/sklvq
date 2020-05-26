import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from . import DistanceBaseClass

from typing import TYPE_CHECKING
from typing import Dict

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass


class SquaredEuclidean(DistanceBaseClass):
    def __init__(self, other_kwargs: Dict = None):
        self.metric_kwargs = {
            "metric": "euclidean",
            "squared": True
        }

        if other_kwargs is not None:
            self.metric_kwargs.update(other_kwargs)

    def __call__(self, data: np.ndarray, model: "LVQBaseClass") -> np.ndarray:
        """ Wrapper function for sklearn pairwise_distances ("euclidean") function

            Parameters
            ----------
            data       : ndarray, shape = [n_obervations, n_features]
                         Inputs are converted to float type.
            prototypes : ndarray, shape = [n_prototypes, n_features]
                         Inputs are converted to float type.

            Returns
            -------
            distances : ndarray, shape = [n_observations, n_prototypes]
                The dist(u=XA[i], v=XB[j]) is computed and stored in the
                ij-th entry.
        """
        return pairwise_distances(
            data,
            model.prototypes_,
            **self.metric_kwargs,
        )

    def gradient(
        self, data: np.ndarray, model: "LVQBaseClass", i_prototype: int
    ) -> np.ndarray:
        """ Implements the derivative of the squared euclidean distance, , with respect to 1 prototype.

            Parameters
            ----------
            model
            data       : ndarray, shape = [n_observations, n_features]

            prototype  : ndarray, shape = [n_features,]

            -------
            gradient : ndarray, shape = [n_observations, n_features]
                        The gradient with respect to the prototype and every observation in data.
        """
        prototypes = model.get_model_params()
        (num_samples, num_features) = data.shape

        distance_gradient = np.zeros((num_samples, prototypes.size))

        ip_start = i_prototype * num_features
        ip_end = ip_start + num_features

        distance_gradient[:, ip_start:ip_end] = -2 * (data - prototypes[i_prototype, :])

        return distance_gradient
