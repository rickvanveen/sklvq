import pytest
import numpy as np

from ..identity import Identity
from ..sigmoid import Sigmoid
from ..soft_plus import SoftPlus
from ..swish import Swish

from sklvq.misc.common_checks import has_call_method, has_gradient_method


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
