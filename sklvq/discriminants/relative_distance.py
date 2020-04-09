from . import DiscriminativeBaseClass
import numpy as np


class RelativeDistance(DiscriminativeBaseClass):
    """ Relative distance function

    """
    def __call__(self, dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
        """ The relative distance discriminant function:
            .. math::

                \\mu(x) = \\frac{d(x, w_1) - d(x, w_2)}{d(x, w_1) + d(x, w_2)},

        with :math:`w_1` the prototype with the same label and :math:`w_2` the prototype with a different label.

        Parameters
        ----------
        dist_same : numpy.ndarray with shape (n_samples, 1)
            The distance from at least one sample to the closest prototype with the same label.
        dist_diff : numpy.ndarray with shape (n_samples, 1)
            The distance from at least one sample to the closest prototype with a different label.

        Returns
        -------
        numpy.ndarray
            The relative distance for each sample

        """
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # Suppresses runtime warning TODO: fix
            return (dist_same - dist_diff) / (dist_same + dist_diff)

    # TODO: Maybe winner_same should be an array that indicates if its the same or different label for every sample...
    def gradient(
        self, dist_same: np.ndarray, dist_diff: np.ndarray, winner_same: bool
    ) -> np.ndarray:
        """ The partial derivative of the relative distance function.

        Two possible scenarios that change based on winner_same being True or False:

            1. The partial derivative with respect to the closest prototypes with the same label (winner_same=True):

            .. math::

                \\frac{\\partial \\mu}{\\partial w_1} = \\frac{2 \\cdot d(x, w_2))}{(d(x, w_1) + d(x, w_2))^2}.

            2. The partial derivative with respect to the closest prototypes with a different label (winner_same=False):

            .. math::

                \\frac{\\partial \\mu}{\\partial w_2} = \\frac{-2 \\cdot d(x, w_1))}{(d(x, w_1) + d(x, w_2))^2},

            with :math:`d(x, w_1)` the distance to the prototype with the same label and :math:`d(x, w_2)` the distance
            to the closest prototype with a different label.

        Parameters
        ----------
        dist_same : numpy.ndarray with shape (n_samples, 1)
            The distance from at least one sample to the closest prototype with the same label.
        dist_diff : numpy.ndarray with shape (n_samples, 1)
            The distance from at least one sample to the closest prototype with a different label.
        winner_same : bool
            Indicating if the derivative with respect to a prototype with the same label (True) or a different label
            (False) needs to be calculated.

        Returns
        -------
        numpy.ndarray
            The relative distance per sample

        """
        if winner_same:
            return _gradient_same(dist_same, dist_diff)
        return _gradient_diff(dist_same, dist_diff)


def _gradient_same(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):  # Suppresses runtime warning
        return 2 * dist_diff / (dist_same + dist_diff) ** 2


def _gradient_diff(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):  # Suppresses runtime warning
        return -2 * dist_same / (dist_same + dist_diff) ** 2
