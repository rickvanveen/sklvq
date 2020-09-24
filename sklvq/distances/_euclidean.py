import numpy as np

from scipy.spatial.distance import cdist

from ._base import DistanceBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GLVQ


class Euclidean(DistanceBaseClass):
    """ Euclidean distance function

    See also
    --------
    SquaredEuclidean, AdaptiveSquaredEuclidean, LocalAdaptiveSquaredEuclidean
    """

    __slots__ = "force_all_finite"

    def __init__(self, force_all_finite=True):
        self.force_all_finite = force_all_finite

    def __call__(self, data: np.ndarray, model: "GLVQ") -> np.ndarray:
        """ Computes the Euclidean distance:
            .. math::

                d(\\vec{w}, \\vec{x}) = \\sqrt{(\\vec{x} - \\vec{w})^{\\top} (\\vec{x} - \\vec{w})},

            with :math:`\\vec{w}` a prototype and :math:`\\vec{x}` a sample.

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The X for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GLVQ
            The model instance.

        Returns
        -------
        ndarray with shape (n_samples, n_prototypes)
            Evaluation of the distance between each sample and prototype of the model.
        """
        distance_function = "euclidean"
        # if self.force_all_finite == "allow-nan" or False:
        #     distance_function = lambda u, v: np.nansum(u - v)**2

        return cdist(data, model.prototypes_, distance_function)

    def gradient(self, data: np.ndarray, model: "GLVQ", i_prototype: int) -> np.ndarray:
        """ Implements the derivative of the squared euclidean distance, with respect to a single
        prototype for the euclidean and nan_euclidean distance:

            .. math::
                \\frac{\\partial d}{\\partial \\vec{w_i}} = -2 \\cdot (\\vec{x} - \\vec{w_i})

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            The X for which the distance gradient to the prototypes of the model need to be
            computed.
        model : GLVQ
            The model instance.
        i_prototype : int
            Index of the prototype to compute the gradient for.

        Returns
        -------
        gradient : ndarray with shape (n_samples, n_features)
            The gradient with respect to the prototype and every sample in the X.

        """
        prototype = model.get_model_params()[i_prototype, :]

        # Compute gradient
        difference = data - prototype

        # TODO nan check

        # Euclidean distance but only to single prototype. Equal to:
        #       np.sqrt(np.sum((X - prototype)**2))
        denominator = np.sqrt(np.einsum("ij, ij -> i", difference, difference))
        # If X is exactly equal to prototype the denominator will be zero. This happens mostly
        # when nan differences are replaced by zero distance. It works.
        denominator[denominator == 0.0] = 1.0

        distance_gradient = -1 * (difference) / denominator[:, np.newaxis]

        # Return 1d array (the original memory)
        return distance_gradient
