from . import DiscriminativeBaseClass
import numpy as np


class RelativeDistance(DiscriminativeBaseClass):
    """ Relative distance function

    """

    def __call__(self, dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
        """ The relative distance discriminant function:
            .. math::

                \\mu(x) = \\frac{d(x, w_1) - d(x, w_0)}{d(x, w_1) + d(x, w_0)},

        with :math:`w_1` the prototype with the same label and :math:`w_0` the prototype with a
        different label.

        Parameters
        ----------
        dist_same : ndarray with shape (n_samples, 1)
            Shortest distance of one or more samples to a prototype with the same label.
        dist_diff : ndarray with shape (n_samples, 1)
            Shortest distance of one or more samples to a prototype with a different label.

        Returns
        -------
        ndarray with shape (n_sampeles, 1)
                Per sample evaluation of the relative distance discriminative function.

        """
        return (dist_same - dist_diff) / (dist_same + dist_diff)

    def gradient(
        self, dist_same: np.ndarray, dist_diff: np.ndarray, same_label: bool
    ) -> np.ndarray:
        """ The partial derivative of the relative distance function.

        Two possible scenarios that change based on same_label being True or False:

            1. The partial derivative with respect to the closest prototypes with the same
                label (same_label=True):

            .. math::

                \\frac{\\partial \\mu}{\\partial w_1} = \\frac{2 \\cdot d(x, w_0))}{(d(x,
                w_1) + d(x, w_0))^2}.

            2. The partial derivative with respect to the closest prototypes with a different
                label (same_label=False):

            .. math::

                \\frac{\\partial \\mu}{\\partial w_0} = \\frac{-2 \\cdot d(x, w_1))}{(d(x,
                w_1) + d(x, w_0))^2},

            with :math:`d(x, w_1)` the distance to the prototype with the same label
            and :math:`d(x, w_0)` the distance to the closest prototype with a different label.

        Parameters
        ----------
        dist_same : ndarray with shape (n_samples, 1)
            Shortest distance of one or more samples to a prototype with the same label.
        dist_diff : ndarray with shape (n_samples, 1)
            Shortest distance of one or more samples to a prototype with a different label.
        same_label : bool
            Indicating if the derivative with respect to a prototype with the same label (True) or
            a different label (False) needs to be calculated.

        Returns
        -------
        ndarray
            Per sample evaluation of the relative distance discriminative function's gradient.

        """
        if same_label:
            return _gradient_same(dist_same, dist_diff)
        return _gradient_diff(dist_same, dist_diff)


def _gradient_same(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    return 2 * dist_diff / (dist_same + dist_diff) ** 2


def _gradient_diff(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    return -2 * dist_same / (dist_same + dist_diff) ** 2
