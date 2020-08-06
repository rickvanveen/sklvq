import pytest
import numpy as np

from sklvq.discriminants import RelativeDistance

from sklvq.misc.common_checks import has_call_method, has_gradient_method


@pytest.mark.parametrize("discriminant", [RelativeDistance])
def test_discriminants(discriminant):
    return check_discriminant(discriminant)


def check_discriminant(discriminant_class):
    # Check compatibility
    has_call_method(discriminant_class)
    has_gradient_method(discriminant_class)

    # Instantiate callable object
    discriminant_callable = discriminant_class()

    # Generate random X
    rng = np.random.RandomState(0)
    x = rng.random_sample((4, 3))
    y = rng.random_sample((4, 3))

    # Check output shape __call__
    assert discriminant_callable(x, y).shape == x.shape

    # Check output type __call__
    assert type(discriminant_callable(x, y)) == np.ndarray

    # Check output shape gradient
    assert discriminant_callable.gradient(x, y, True).shape == x.shape
    assert discriminant_callable.gradient(x, y, False).shape == x.shape

    # Check output type gradient
    assert type(discriminant_callable.gradient(x, y, True)) == np.ndarray
    assert type(discriminant_callable.gradient(x, y, False)) == np.ndarray
