from ._base import DiscriminantBaseClass
import numpy as np


class RelativeDistance(DiscriminantBaseClass):
    """Relative distance function

    Class that holds the relative distance function and gradient as described in `[1]`_.

    References
    ----------
    _`[1]` Sato, A., and Yamada, K. (1996) "Generalized Learning Vector Quantization."
    Advances in Neural Network Information Processing Systems, 423–429, 1996."""

    __slots__ = ()

    def __call__(self, dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
        r"""The relative distance discriminant function for a single sample (:math:`\mathbf{x}`):
            .. math::

                \mu(\mathbf{x}) = \frac{d(\mathbf{x}, \mathbf{w}_1) - d(\mathbf{x}, \mathbf{w}_0)}{d(
                \mathbf{x}, \mathbf{w}_1) + d(\mathbf{x}, \mathbf{w}_0)},

        with :math:`\mathbf{w}_1` the prototype with the same label and :math:`\mathbf{w}_0` the
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
        r"""Computes the relative distance discriminant function's gradient.

            1. The partial derivative with respect to the closest prototypes with the same label (same_label=True):

            .. math::

                \frac{\partial \mu}{\partial \mathbf{w}_1} = \frac{2 \cdot d(\mathbf{x},
                \mathbf{w}_0))}{(d(\mathbf{x}, \mathbf{w}_1) + d(\mathbf{x}, \mathbf{w}_0))^2}.

            2. The partial derivative with respect to the closest prototypes with a different label (same_label=False):

            .. math::

                \frac{\partial \mu}{\partial \mathbf{w}_0} = \frac{-2 \cdot d(\mathbf{x},
                \mathbf{w}_1))}{(d(\mathbf{x}, \mathbf{w}_1) + d(\mathbf{x}, \mathbf{w}_0))^2},

            with :math:`d(\mathbf{x}, \mathbf{w}_1)` the distance to the prototype with the same label
            and :math:`d(\mathbf{x}, \mathbf{w}_0)` the distance to the closest prototype with a
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
        ndarray with shape (n_samples, 1)
            Evaluation of the relative distance function's gradient.

        """
        if same_label:
            return _gradient_same(dist_same, dist_diff)
        return _gradient_diff(dist_same, dist_diff)


def _gradient_same(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    return 2 * dist_diff / (dist_same + dist_diff) ** 2


def _gradient_diff(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    return -2 * dist_same / (dist_same + dist_diff) ** 2
