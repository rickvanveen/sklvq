from __future__ import annotations

import numpy as np

from sklvq.activations._base import ActivationBaseClass


class SoftPlus(ActivationBaseClass):
    """Soft+ function

    Class that holds the soft+ function and gradient as discussed in `[1]`_

    Parameters
    ----------
    beta : int or float, optional, default=1
           Positive non-zero value that controls the steepness of the Soft+ function.

    See also
    --------
    Identity, Sigmoid, Swish

    References
    ----------
    _`[1]` Villmann, T., Ravichandran, J., Villmann, A., Nebel, D., & Kaden, M. (2019). "Activation
    Functions for Generalized Learning Vector Quantization - A Performance Comparison", 2019.
    """

    __slots__: tuple[str, ...] = ("beta",)

    def __init__(self, beta: float = 1):
        if beta <= 0:
            msg = f"{type(self).__name__}: Expected beta > 0, but got beta = {beta}"
            raise ValueError(msg)

        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r"""Implements the soft+ function:
            .. math::

                f(\mathbf{x}) = \ln(1 + e^{\beta \cdot \mathbf{x}})

        Parameters
        ----------
        x : ndarray of any shape

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of the soft+ function.
        """
        return np.log(1 + np.exp(self.beta * x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        r"""Implements the sigmoid function's gradient:
            .. math::

                \frac{\partial f}{\partial \mathbf{x}} = \frac{\beta \cdot e^{\beta \cdot \mathbf{x}}}{1 + e^{\beta \cdot \mathbf{x}}}

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of the soft+ function's gradient.
        """

        exp = np.exp(self.beta * x)
        return (self.beta * exp) / (1 + exp)
