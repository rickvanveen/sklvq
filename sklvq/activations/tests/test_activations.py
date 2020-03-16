import pytest
import numpy as np

from sklvq import activations
from sklvq.activations.identity import Identity
from sklvq.activations.sigmoid import Sigmoid
from sklvq.activations.soft_plus import SoftPlus
from sklvq.activations.swish import Swish

from sklearn.utils._testing import assert_array_almost_equal


# values producing nice round results
# inflection points
# relative and absolute borders


# Check if grab returns correct class, defaults of init (if any), basic workings are correct.
def test_identity():
    identity = activations.grab('identity', None)
    # Test if grab returns the correct class
    assert isinstance(identity, Identity)

    # Random data - which is okay for the identity function
    rng = np.random.RandomState(0)
    x = rng.random_sample((5, 4))

    # Workings of identity are pretty simple
    assert np.all(identity(x) == x)
    assert np.all(identity.gradient(x) == np.ones(x.shape))


def test_sigmoid():
    default_beta = 1
    other_beta = 10

    # Check if defaults are set using grab method
    sigmoid = activations.grab('sigmoid', None)
    assert isinstance(sigmoid, Sigmoid)
    assert sigmoid.beta == default_beta

    assert (sigmoid(np.array([0])) == pytest.approx(0.5))
    assert (sigmoid.gradient(np.array([0])) == pytest.approx(0.25))

    # Always positive
    assert (sigmoid(np.array([-1]) > 0))
    assert (sigmoid(np.array([1]) > 0))

    # Symmetry
    assert (1 - sigmoid(np.array([-1])) == pytest.approx(sigmoid(np.array([1]))))

    # Check if parameters are passed to sigmoid class when using grab
    sigmoid = activations.grab('sigmoid', {'beta': other_beta})
    assert isinstance(sigmoid, Sigmoid)
    assert sigmoid.beta == other_beta

    assert (sigmoid(np.array([0])) == pytest.approx(0.5))
    assert (sigmoid.gradient(np.array([0])) == pytest.approx(2.5))


def test_soft_plus():
    default_beta = 1
    other_beta = 10

    soft_plus = activations.grab('soft+', None)
    assert isinstance(soft_plus, SoftPlus)
    assert soft_plus.beta == default_beta

    soft_plus = activations.grab('soft-plus', None)
    assert isinstance(soft_plus, SoftPlus)
    assert soft_plus.beta == default_beta

    soft_plus = activations.grab('soft+', {'beta': other_beta})
    assert isinstance(soft_plus, SoftPlus)
    assert soft_plus.beta == other_beta

    soft_plus = activations.grab('soft-plus', {'beta': other_beta})
    assert isinstance(soft_plus, SoftPlus)
    assert soft_plus.beta == other_beta


def test_swish():
    default_beta = 1
    other_beta = 10

    swift = activations.grab('swish', None)
    assert isinstance(swift, Swish)
    assert swift.beta == default_beta

    swift = activations.grab('swish', {'beta': other_beta})
    assert isinstance(swift, Swish)
    assert swift.beta == other_beta
