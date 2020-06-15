import pytest
import numpy as np

from sklvq.activations.identity import Identity
from sklvq.activations.sigmoid import Sigmoid
from sklvq.activations.soft_plus import SoftPlus
from sklvq.activations.swish import Swish

from sklvq import activations

from sklvq.misc.common_checks import has_call_method, has_gradient_method
from sklearn.utils._testing import assert_array_equal


@pytest.mark.parametrize(
    "activation", [Identity, Sigmoid, SoftPlus, Swish]
)
def test_activations(activation):
    return check_activation(activation)


def check_activation(activation_class):
    # Check compatibility
    has_call_method(activation_class)
    has_gradient_method(activation_class)

    # Instantiate callable object
    activation_callable = activation_class()

    # Generate random data
    rng = np.random.RandomState(0)
    x = rng.random_sample((3, 4))

    # Check output shape __call__
    assert activation_callable(x).shape == x.shape

    # Check output type __call__
    assert type(activation_callable(x)) == np.ndarray

    # Check output shape gradient
    assert activation_callable.gradient(x).shape == x.shape

    # Check output type gradient
    assert type(activation_callable.gradient(x)) == np.ndarray