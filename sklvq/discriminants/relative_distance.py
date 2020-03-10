from . import DiscriminativeBaseClass
import numpy as np


class RelativeDistance(DiscriminativeBaseClass):

    def __call__(self, dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
        """ Implements the relative distance discriminant function :cite:`Sato1996`
            .. math::

                \\mu(x) = \\frac{d(x, w_1) - d(x, w_2)}{d(x, w_1) + d(x, w_2)},

            with :math:`w_1` the prototype with the same label and :math:`w_2` the prototype with a different label.

        Parameters
        ----------
        dist_same : ndarray
            The distance from at least one sample to the closest prototype with the same label.
        dist_diff : ndarray
            The distance from at least one sample to the closest prototype with a different label.

        Returns
        -------
        ndarray
            The relative distance per sample

        """
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning TODO: fix
            return (dist_same - dist_diff) / (dist_same + dist_diff)

    def gradient(self, dist_same: np.ndarray, dist_diff: np.ndarray, winner_same: bool) -> np.ndarray:
        """ The partial derivative of the discriminant function with respect to the winning prototypes with the same
        or different label.

        Parameters
        ----------
        dist_same : ndarray
            The distance from at least one sample to the closest prototype with the same label.
        dist_diff : ndarray
            The distance from at least one sample to the closest prototype with a different label.
        winner_same : bool
            Indicating if the derivative with respect to a prototype with the same label (True) or a different label
            (False) needs to be calculated.

        Returns
        -------
        ndarray
            The relative distance per sample

        """
        if winner_same:
            return _gradient_same(dist_same, dist_diff)
        return _gradient_diff(dist_same, dist_diff)


def _gradient_same(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    """ The partial derivative of the discriminant function with respect to the prototype with the same label:
        .. math::

            \\frac{\\partial \\mu}{\\partial w_1} = \\frac{2 \\cdot d(x, w_1))}{(d(x, w_1) + d(x, w_2))^2},

        with :math:`w_1` the prototype with the same label and :math:`w_2` the prototype with a different label.

    Parameters
    ----------
    dist_same : ndarray
        The distance from at least one sample to the closest prototype with the same label.
    dist_diff : ndarray
        The distance from at least one sample to the closest prototype with a different label.

    Returns
    -------
    ndarray
        The relative distance per sample

    """
    with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
        return 2 * dist_diff / (dist_same + dist_diff) ** 2


def _gradient_diff(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    """ The partial derivative of the discriminant function with respect to the prototype with a different label:
        .. math::

            \\frac{\\partial \\mu}{\\partial w_2} = \\frac{-2 \\cdot d(x, w_1))}{(d(x, w_1) + d(x, w_2))^2},

        with :math:`w_1` the prototype with the same label and :math:`w_2` the prototype with a different label.

    Parameters
    ----------
    dist_same : ndarray
        The distance from at least one sample to the closest prototype with the same label.
    dist_diff : ndarray
        The distance from at least one sample to the closest prototype with a different label.

    Returns
    -------
    ndarray
        The relative distance per sample

    """
    with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
        return -2 * dist_same / (dist_same + dist_diff) ** 2
