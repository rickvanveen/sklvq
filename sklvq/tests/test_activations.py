import pytest

from sklvq import activations
from sklvq.activations.identity import Identity
from sklvq.activations.sigmoid import Sigmoid
from sklvq.activations.soft_plus import SoftPlus
from sklvq.activations.swish import Swish
from sklvq.activations.base import ActivationBaseClass
import numpy as np


@pytest.mark.parametrize(
    "callable_class", [Identity, Sigmoid, SoftPlus, Swish]
)
def test_activation_compatibility(callable_class):
    # Check if the callable is a subclass of the activation base class or at least implements the two required
    # functions.
    assert (issubclass(callable_class, ActivationBaseClass)
            | (hasattr(callable_class, '__call__') & hasattr(callable_class, 'gradient')))

    # Random data
    size = 12
    shape = (3,4)
    x = np.random.rand(size).reshape(shape)

    # Instantiate callable object
    instance = callable_class()

    # Some size and shape checks
    assert ((instance(x).size == size)
            & (instance(x).shape == shape))
    assert ((instance.gradient(x).size == size)
            & (instance.gradient(x).shape == shape))

    assert type(instance(x)) == np.ndarray
    assert type(instance.gradient(x)) == np.ndarray


@pytest.mark.parametrize(
    "string", ['identity', 'sigmoid', 'soft-plus', 'swish']
)
def test_activations_init(string):
    activations.grab(string, None)


def test_identity():
    identity = Identity()

    # Random data
    size = 12
    shape = (3,4)
    x = np.random.rand(size).reshape(shape)

    assert np.all(identity(x) == x)
    assert np.all(identity.gradient(x) == np.ones(x.shape))


def test_sigmoid():

    # Check init
    sigmoid = activations.grab('sigmoid', {'beta': 6})
    assert sigmoid.beta == 6

    # Check values (random values)
    x = np.array([-1.2, -1, 0, 1, 1.2])
    assert sigmoid(x) == pytest.approx([
        7.46028834e-04, 2.47262316e-03,
        5.00000000e-01, 9.97527377e-01,
        9.99253971e-01])

    # Check default value
    sigmoid = activations.grab('sigmoid', None)
    assert sigmoid.beta == 1

    # Check values (random values)
    x = np.array([-1.2, -1, 0, 1, 1.2])
    assert sigmoid(x) == pytest.approx([
        0.23147522, 0.26894142,
        0.5, 0.73105858,
        0.76852478])
