import numpy as np

from scipy.spatial.distance import cdist

from ._base import DistanceBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GLVQ


class Euclidean(DistanceBaseClass):
    """Euclidean distance function

    Class that holds the euclidean distance function and its gradient.

    Parameters
    ----------
    force_all_finite  : {True, False, "allow-nan"}
        Parameter to indicate that NaNLVQ distance variant should be used. If true no nans are
        allowed. If False or "allow-nan" nans are allowed.

    See also
    --------
    SquaredEuclidean, AdaptiveSquaredEuclidean, LocalAdaptiveSquaredEuclidean

    Notes
    -----
    Compatible with the :class:`.GLVQ` algorithm (only).
    """

    __slots__ = ()

    def __call__(self, data: np.ndarray, model: "GLVQ") -> np.ndarray:
        r"""Computes the Euclidean distance:
            .. math::

                d(\mathbf{w}, \mathbf{x}) = \sqrt{(\mathbf{x} - \mathbf{w})^{\top} (\mathbf{x} - \mathbf{w})},

            with :math:`\mathbf{w}` a prototype and :math:`\mathbf{x}` a sample.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distances to the prototypes of the model need to be
            computed.
        model : GLVQ
            A GLVQ model instance, containing the prototypes.

        Returns
        -------
        ndarray with shape (n_samples, n_prototypes)
            Evaluation of the distance between each sample and prototype of the model.
        """
        distance_function = "euclidean"
        if model.force_all_finite == "allow-nan":
            distance_function = _nan_euclidean

        return cdist(data, model.prototypes_, distance_function)

    def gradient(self, data: np.ndarray, model: "GLVQ", i_prototype: int) -> np.ndarray:
        r"""Computes the gradient of the euclidean distance with respect to a single
        prototype:

            .. math::
                \frac{\partial d}{\partial \mathbf{w}_i} = -2 \cdot (\mathbf{x} - \mathbf{w}_i)

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GLVQ
            A GLVQ model instance.
        i_prototype : int
            Index of the prototype to compute the gradient for.

        Returns
        -------
        ndarray with shape (n_samples, n_features)
            The gradient of the prototype with respect to every sample in the data.
        """
        prototype = model.get_model_params()[i_prototype, :]

        difference = data - prototype

        if model.force_all_finite == "allow-nan" or False:
            difference[np.isnan(difference)] = 0.0

        # Euclidean distance to single prototype. Equal to: np.sqrt(np.sum((data - prototype)**2))
        denominator = np.sqrt(np.einsum("ij, ij -> i", difference, difference))

        # Might happen if a sample is exactly on a prototype, mostly caused by nans in the data.
        denominator[denominator == 0.0] = 1.0

        distance_gradient = -1 * difference / denominator[:, np.newaxis]

        return distance_gradient


def _nan_euclidean(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Euclidean distance between two vectors u and v, ignoring nans.
    return np.sqrt(np.nansum(u - v) ** 2)
