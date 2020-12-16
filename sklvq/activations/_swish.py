from . import ActivationBaseClass
import numpy as np

from typing import Union


class Swish(ActivationBaseClass):
    """Swish function

    Class that holds the swish function and gradient as discussed in `[1]`_

    Parameters
    ----------
    beta : int, float, default=1
          Positive non-zero value that controls the steepness of the Swish function.

    See also
    --------
    Identity, Sigmoid, SoftPlus

    References
    ----------
    _`[1]` Villmann, T., Ravichandran, J., Villmann, A., Nebel, D., & Kaden, M. (2019). "Activation
    Functions for Generalized Learning Vector Quantization - A Performance Comparison", 2019.
    """

    __slots__ = "beta"

    def __init__(self, beta: Union[int, float] = 1):
        if beta <= 0:
            raise ValueError(
                "The activation function {} expects beta > 0, but got beta = {}".format(
                    type(self).__name__, beta
                )
            )

        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r"""Implements the swish function:
            .. math::

                f(\mathbf{x}) = \frac{\mathbf{x}}{1 + e^{-\beta \cdot \mathbf{x}}}

        Parameters
        ----------
        x : ndarray of any shape

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of the swish function.
        """
        return _swish(x, self.beta)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        r"""Implements the sigmoid function's gradient:
            .. math::

                \frac{\partial f}{\partial \mathbf{x}} = \beta \cdot f(\mathbf{x}) + (\frac{1}{1 + e^{-\beta \cdot \mathbf{x}}}) \cdot (1 - \beta \cdot f(\mathbf{x}))

        Parameters
        ----------
        x : ndarray of any shape

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of the swish function's gradient.
        """
        return (self.beta * _swish(x, self.beta)) + (
            _sgd(x, self.beta) * (1 - self.beta * _swish(x, self.beta))
        )


def _sgd(x: np.ndarray, beta: int) -> np.ndarray:
    # Sigmoid function
    return 1 / (np.exp(-beta * x) + 1)


def _swish(x: np.ndarray, beta: int) -> np.ndarray:
    # Swish function
    return x * _sgd(x, beta)
