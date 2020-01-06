import scipy as sp
import numpy as np

from . import DistanceBaseClass


class Euclidean(DistanceBaseClass):

    def __call__(self, data: np.ndarray, model) -> np.ndarray:
        """ Wrapper function for scipy's cdist(x, y, 'euclidean') function

            See scipy.spatial.distance.cdist for full documentation.

            Note that any custom function should still accept and return the same.

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
        return sp.spatial.distance.cdist(data, model.prototypes_, 'euclidean')

    def gradient(self, data: np.ndarray, model, i_prototype: int) -> np.ndarray:
        """ Implements the derivative of the euclidean distance, with respect to 1 prototype

            Parameters
            ----------

            Returns
            -------
            gradient : ndarray, shape = [n_observations, n_features]
                       The gradient with respect to the prototype and every observation in data.

        """
        shape = [data.shape[0], *model.prototypes_.shape]
        gradient = np.zeros(shape)

        difference = data - model.prototypes_[i_prototype, :]

        gradient[:, i_prototype, :] = np.atleast_2d(
            (-1 * difference) / np.sqrt(np.sum(difference ** 2)))

        return gradient.reshape(shape[0], shape[1] * shape[2])