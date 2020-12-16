import numpy as np

from scipy.spatial.distance import cdist

from ._base import DistanceBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GLVQ


class SquaredEuclidean(DistanceBaseClass):
    """Squared Euclidean distance

    Parameters
    ----------
    force_all_finite  : {True, False, "allow-nan"}
        Parameter to indicate that NaNLVQ distance variant should be used. If true no nans are
        allowed. If False or "allow-nan" nans are allowed.

    See also
    --------
    Euclidean, AdaptiveSquaredEuclidean, LocalAdaptiveSquaredEuclidean

    Notes
    -----
    Compatible with the :class:`.GLVQ` algorithm (only).
    """

    __slots__ = ()

    def __call__(self, data: np.ndarray, model: "GLVQ") -> np.ndarray:
        r"""
         Computes the squared Euclidean distance:
            .. math::

                d(\mathbf{w}, \mathbf{x}) = (\mathbf{x} - \mathbf{w})^{\top} (\mathbf{x} - \mathbf{w}),

            with :math:`\mathbf{w}` a prototype and :math:`\mathbf{x}` a sample.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GLVQ
            The GLVQ model instance, containing the prototypes.

        Returns
        -------
        ndarray with shape (n_samples, n_prototypes)
            Evaluation of the distance between each sample and prototype of the model.
        """
        distance_function = "sqeuclidean"
        if model.force_all_finite == "allow-nan":
            distance_function = _nan_squared_euclidean

        return cdist(data, model.prototypes_, distance_function)

    def gradient(self, data: np.ndarray, model: "GLVQ", i_prototype: int) -> np.ndarray:
        r"""Computes the gradient of the squared euclidean distance, with respect to a single
        prototype:

            .. math::
                \frac{\partial d}{\partial \mathbf{w}_i} = -2 \cdot (\mathbf{x} - \mathbf{w}_i)

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The data for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GLVQ
            The GLVQ model instance.
        i_prototype : int
            Index of the prototype to compute the gradient for.

        Returns
        -------
        gradient : ndarray with shape (n_samples, n_features)
            The gradient of the prototype with respect to every sample in the data.
        """
        distance_gradient = -2 * (data - model.get_model_params()[i_prototype, :])

        # In case of nans replace nan values by 0.0
        if model.force_all_finite == "allow-nan":
            distance_gradient[np.isnan(distance_gradient)] = 0.0

        # Return 1d array (the original memory)
        return distance_gradient


def _nan_squared_euclidean(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Squared Euclidean distance between two vectors u and v, ignoring nans.
    return np.nansum(u - v) ** 2
