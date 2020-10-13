from ._base import DiscriminativeBaseClass
import numpy as np


class RelativeDistance(DiscriminativeBaseClass):
    """ Relative distance function

    Class that holds the relative distance function and gradient.

    """
    __slots__ = ()

    def __call__(self, dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
        """ The relative distance discriminant function for a single sample (:math:`\\vec{x}`):
            .. math::

                \\mu(\\vec{x}) = \\frac{d(\\vec{x}, \\vec{w}_1) - d(\\vec{x}, \\vec{w}_0)}{d(
                \\vec{x}, \\vec{w}_1) + d(\\vec{x}, \\vec{w}_0)},

        with :math:`\\vec{w}_1` the prototype with the same label and :math:`\\vec{w}_0` the
        prototype with a different label.

        Parameters
        ----------
        dist_same : ndarray with shape (n_samples, 1), with n_samples >= 1
            Shortest distance of n_samples to a prototype with the same label.
        dist_diff : ndarray with shape (n_samples, 1), with n_samples >= 1
            Shortest distance of n_samples to a prototype with a different label.

        Returns
        -------
        ndarray with shape (n_samples, 1)
            Evaluation of the relative distance discriminative function.

        """
        return (dist_same - dist_diff) / (dist_same + dist_diff)

    def gradient(
        self, dist_same: np.ndarray, dist_diff: np.ndarray, same_label: bool
    ) -> np.ndarray:
        """ The relative distance discriminant function's gradient.

            1. The partial derivative with respect to the closest prototypes with the same label (same_label=True):

            .. math::

                \\frac{\\partial \\mu}{\\partial \\vec{w}_1} = \\frac{2 \\cdot d(\\vec{x},
                \\vec{w}_0))}{(d(\\vec{x}, \\vec{w}_1) + d(\\vec{x}, \\vec{w}_0))^2}.

            2. The partial derivative with respect to the closest prototypes with a different label (same_label=False):

            .. math::

                \\frac{\\partial \\mu}{\\partial \\vec{w}_0} = \\frac{-2 \\cdot d(\\vec{x},
                \\vec{w}_1))}{(d(\\vec{x}, \\vec{w}_1) + d(\\vec{x}, \\vec{w}_0))^2},

            with :math:`d(\\vec{x}, \\vec{w}_1)` the distance to the prototype with the same label
            and :math:`d(\\vec{x}, \\vec{w}_0)` the distance to the closest prototype with a
            different label.

        Parameters
        ----------
        dist_same : ndarray with shape (n_samples, 1), with n_samples >= 1
            Shortest distance of n_samples to a prototype with the same label.
        dist_diff : ndarray with shape (n_samples, 1), with n_samples >= 1
            Shortest distance of n_samples to a prototype with a different label.
        same_label : bool
            Indicating if the derivative with respect to a prototype with the same label (True) or
            a different label (False) needs to be calculated.

        Returns
        -------
        ndarray
            Evaluation of the relative distance function's gradient.

        """
        if same_label:
            return _gradient_same(dist_same, dist_diff)
        return _gradient_diff(dist_same, dist_diff)


def _gradient_same(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    return 2 * dist_diff / (dist_same + dist_diff) ** 2


def _gradient_diff(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    return -2 * dist_same / (dist_same + dist_diff) ** 2
