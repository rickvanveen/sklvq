import pytest
import numpy as np

from sklvq import activations
from sklvq.activations.identity import Identity
from sklvq.activations.sigmoid import Sigmoid
from sklvq.activations.soft_plus import SoftPlus
from sklvq.activations.swish import Swish

from sklearn.utils._testing import assert_array_almost_equal


# Check if grab returns correct class, defaults of init (if any), basic workings are correct.
def test_identity():
    identity = activations.grab('identity', None)
    # Test if grab returns the correct class
    assert isinstance(identity, Identity)

    identity = Identity()

    # Random data
    rng = np.random.RandomState(0)
    x = rng.random_sample((5, 4))

    # Workings of identity are pretty simple
    assert np.all(identity(x) == x)
    assert np.all(identity.gradient(x) == np.ones(x.shape))


def test_sigmoid():
    default_beta = 1
    other_beta = 6

    rng = np.random.RandomState(0)
    x = rng.random_sample((5, 4))

    # Check defaults
    sigmoid = activations.grab('sigmoid', None)
    assert isinstance(sigmoid, Sigmoid)
    assert sigmoid.beta == default_beta
    s1 = sigmoid(x)
    g1 = sigmoid.gradient(x)

    sigmoid = Sigmoid()
    assert sigmoid.beta == default_beta
    s2 = sigmoid(x)
    g2 = sigmoid.gradient(x)

    # Check if grab and class() give same answer
    assert_array_almost_equal(s1, s2)
    assert_array_almost_equal(g1, g2)

    # Check other init
    sigmoid = activations.grab('sigmoid', {'beta': other_beta})
    assert isinstance(sigmoid, Sigmoid)
    assert sigmoid.beta == other_beta
    s1 = sigmoid(x)
    g1 = sigmoid(x)

    sigmoid = Sigmoid(beta=other_beta)
    assert sigmoid.beta == other_beta
    s2 = sigmoid(x)
    g1 = sigmoid(x)

    # Check if grab and class() give same answer
    assert_array_almost_equal(s1, s2)
    assert_array_almost_equal(g1, g2)


def test_soft_plus():
    default_beta = 1
    other_beta = 6

    rng = np.random.RandomState(0)
    x = rng.random_sample((5, 4))

    soft_plus = activations.grab('soft+', None)
    assert isinstance(soft_plus, SoftPlus)
    assert soft_plus.beta == default_beta
    s1 = soft_plus(x)
    g1 = soft_plus.gradient(x)

    soft_plus = activations.grab('soft-plus', None)
    assert isinstance(soft_plus, SoftPlus)
    assert soft_plus.beta == default_beta
    s2 = soft_plus(x)
    g2 = soft_plus.gradient(x)

    soft_plus = SoftPlus()
    assert soft_plus.beta == default_beta
    s3 = soft_plus(x)
    g3 = soft_plus.gradient(x)

    # Check if they all provide the same answer
    assert_array_almost_equal(s1, s2)
    assert_array_almost_equal(s1, s3)
    assert_array_almost_equal(g1, g2)
    assert_array_almost_equal(g1, g3)

    soft_plus = activations.grab('soft+', {'beta': other_beta})
    assert isinstance(soft_plus, SoftPlus)
    assert soft_plus.beta == other_beta
    s1 = soft_plus(x)
    g1 = soft_plus.gradient(x)

    soft_plus = activations.grab('soft-plus', {'beta': other_beta})
    assert isinstance(soft_plus, SoftPlus)
    assert soft_plus.beta == other_beta
    s2 = soft_plus(x)
    g2 = soft_plus.gradient(x)

    soft_plus = SoftPlus(beta=other_beta)
    assert soft_plus.beta == other_beta
    s3 = soft_plus(x)
    g3 = soft_plus.gradient(x)

    # Check if they all provide the same answer and use beta
    assert_array_almost_equal(s1, s2)
    assert_array_almost_equal(s2, s3)
    assert_array_almost_equal(g1, g2)
    assert_array_almost_equal(g1, g3)


def test_swish():
    default_beta = 1
    other_beta = 6

    rng = np.random.RandomState(0)
    x = rng.random_sample((5, 4))

    swift = activations.grab('swish', None)
    assert isinstance(swift, Swish)
    assert swift.beta == default_beta
    s1 = swift(x)
    g1 = swift(x)

    swish = Swish()
    assert swish.beta == default_beta
    s2 = swish(x)
    g2 = swift(x)

    assert_array_almost_equal(s1, s2)
    assert_array_almost_equal(g1, g2)

    swift = activations.grab('swish', {'beta': other_beta})
    assert isinstance(swift, Swish)
    assert swift.beta == other_beta
    s1 = swift(x)
    g1 = swift(x)

    swish = Swish(beta=other_beta)
    assert swish.beta == other_beta
    s2 = swish(x)
    g2 = swift(x)

    assert_array_almost_equal(s1, s2)
    assert_array_almost_equal(g1, g2)

