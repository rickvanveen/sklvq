import numpy as np
from typing import Union

from . import ActivationBaseClass


class Sigmoid(ActivationBaseClass):
    """Sigmoid function

    Class that holds the sigmoid function and gradient as discussed in `[1]`_

    Parameters
    ----------
    beta : int or float, optional, default=1
           Positive non-zero value that controls the steepness of the Sigmoid function.

    See also
    --------
    Identity, SoftPlus, Swish

    References
    ----------
    _`[1]` Villmann, T., Ravichandran, J., Villmann, A., Nebel, D., & Kaden, M. (2019). "Activation
    Functions for Generalized Learning Vector Quantization - A Performance Comparison", 2019.
    """

    __slots__ = "beta"

    def __init__(self, beta: Union[int, float] = 1):
        if beta <= 0:
            raise ValueError(
                "{}: Expected beta > 0, but got beta = {}".format(
                    type(self).__name__, beta
                )
            )

        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r"""Computes the sigmoid function:
            .. math::

                f(\mathbf{x}) = \frac{1}{e^{-\beta \cdot \mathbf{x}} + 1}

        Parameters
        ----------
        x : ndarray of any shape.

        Returns
        -------
        ndarray of shape (x.shape)
            Elementwise evaluation of the sigmoid function.
        """
        return np.asarray(1.0 / (np.exp(-self.beta * x) + 1.0))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        r"""Computes the sigmoid function's gradient with respect to x:
           .. math::

               \frac{\partial f}{\partial \mathbf{x}} = \frac{(\beta \cdot e^{\beta \cdot \mathbf{x})}}{(e^{\beta \cdot \mathbf{x}} + 1)^2}

        Parameters
        ----------
        x : ndarray of any shape

        Returns
        -------
        ndarray of shape (x.shape)
           Elementwise evaluation of the sigmoid function's gradient.
        """
        exp = np.exp(self.beta * x)
        return np.asarray((self.beta * exp) / (exp + 1.0) ** 2)
