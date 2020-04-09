import scipy as sp
import numpy as np

from . import DistanceBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class Euclidean(DistanceBaseClass):
    """ Euclidean distance function

    See also
    --------
    SquaredEuclidean, AdaptiveEuclidean, AdaptiveSquaredEuclidean
    """

    def __call__(self, data: np.ndarray, model: "LVQClassifier") -> np.ndarray:
        """
        Computes the Euclidean distance:
            .. math::

                d(\\vec{w}, \\vec{x}) = \\sqrt{(\\vec{x} - \\vec{w})^T (\\vec{x} - \\vec{w})},

            with :math:`\\vec{w}` a prototype and :math:`\\vec{x}` a sample.

        .. note::
            Makes use of the scipy's cdist(x, y, 'euclidean') function, see scipy.spatial.distance.cdist
            for full documentation.

        Parameters
        ----------
        data : numpy.ndarray with shape (n_samples, n_features)
        model : LVQClassifier
            Can be any LVQClassifier but only prototypes will be used to compute the distance

        Returns
        -------
        distances : numpy.ndarray with shape (n_observations, n_prototypes)
            The dist(u=XA[i], v=XB[j]) is computed and stored in the
            ij-th entry.
        """
        return sp.spatial.distance.cdist(data, model.prototypes_, "euclidean")

    def gradient(self, data: np.ndarray, model, i_prototype: int) -> np.ndarray:
        """ Implements the derivative of the euclidean distance, with respect to a single prototype

        Parameters
        ----------
        data : numpy.ndarray with shape (n_samples, n_features)
        model : LVQClassifier
            Only prototypes need to be available in the LVQClassifier
        i_prototype : int
            Index of the prototype to compute the gradient for

        Returns
        -------
        gradient : ndarray with shape (n_samples, n_features)
            The gradient with respect to the prototype and every sample in data.

        """
        shape = [data.shape[0], *model.prototypes_.shape]
        gradient = np.zeros(shape)

        difference = data - model.prototypes_[i_prototype, :]

        gradient[:, i_prototype, :] = np.atleast_2d(
            (-1 * difference) / np.sqrt(np.sum(difference ** 2))
        )

        return gradient.reshape(shape[0], shape[1] * shape[2])
