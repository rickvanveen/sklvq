import pytest
import numpy as np

from sklvq import activations
from sklvq.activations.identity import Identity
from sklvq.activations.sigmoid import Sigmoid
from sklvq.activations.soft_plus import SoftPlus
from sklvq.activations.swish import Swish


# Check if grab returns correct class, defaults of init (if any), basic workings are correct.
def test_identity():
    identity = activations.grab("identity", None)
    # Test if grab returns the correct class
    assert isinstance(identity, Identity)

    # Random X - which is okay for the identity function
    rng = np.random.RandomState(0)
    x = rng.random_sample((5, 4))

    # Workings of identity are pretty simple
    assert np.all(identity(x) == x)
    assert np.all(identity.gradient(x) == np.ones(x.shape))


def test_sigmoid():
    default_beta = 1
    other_beta = 10

    # Check if defaults are set using grab method
    sigmoid = activations.grab("sigmoid")
    assert isinstance(sigmoid, Sigmoid)
    assert sigmoid.beta == default_beta

    # Check if x = 0 equals approximately 0.5
    assert sigmoid(np.array([0])) == pytest.approx(0.5)
    # Check if at x = 0 the gradient equals approximately 0.25
    assert sigmoid.gradient(np.array([0])) == pytest.approx(0.25)

    # "Always" positive with negative numbers
    assert np.all(sigmoid(np.array([-1, -10, -100, -1000, -10000]) > 0))
    # "Always" positive with positive numbers
    assert np.all(sigmoid(np.array([1, 10, 100, 1000, 10000]) > 0))

    # Symmetry
    assert np.all(
        pytest.approx(sigmoid(np.array([1, 2, 3, 4])))
        == (1 - sigmoid(np.array([-1, -2, -3, -4])))
    )

    # Check if parameters are passed to sigmoid class when using grab
    sigmoid = activations.grab("sigmoid", class_kwargs={"beta": other_beta})
    assert isinstance(sigmoid, Sigmoid)
    assert sigmoid.beta == other_beta

    # Check if x = 0 is not changed
    assert sigmoid(np.array([0])) == pytest.approx(0.5)
    # Check if x = 0 is changed based on beta.
    assert sigmoid.gradient(np.array([0])) == pytest.approx(2.5)


def test_soft_plus():
    default_beta = 1
    other_beta = 10

    # Check if defaults are set using grab method
    soft_plus = activations.grab("soft-plus", None)
    assert isinstance(soft_plus, SoftPlus)
    assert soft_plus.beta == default_beta

    # >0 at x = 0
    assert soft_plus(np.array([0])) > 0

    # >0 at x < 0
    assert np.all(soft_plus(np.array([-2, -4, -6, -8, -20, -50, -100]) > 0))

    # Increasing upto gradient of beta...
    assert np.max(soft_plus.gradient(np.array([0, 4, 8, 12, 20]))) == pytest.approx(
        soft_plus.beta
    )

    soft_plus = activations.grab("soft-plus", class_kwargs={"beta": other_beta})
    assert isinstance(soft_plus, SoftPlus)
    assert soft_plus.beta == other_beta

    # Increasing upto gradient of beta...
    assert np.max(soft_plus.gradient(np.array([0, 4, 8, 12, 20]))) == pytest.approx(
        soft_plus.beta
    )

    # soft plus gradient becomes nan when 1 + (beta * e^x) becomes inf. Which happens with larger x.


def test_swish():
    default_beta = 1
    other_beta = 10

    swish = activations.grab("swish", None)
    assert isinstance(swish, Swish)
    assert swish.beta == default_beta

    # x = 0 should be 0
    assert swish(np.array([0])) == 0
    # x < 0 is negative and goes to 0
    assert swish(np.array([-1])) < 0

    # Swish is approximately the identity function, when x gets large enough, thus gradient is 1.
    x = np.array([4, 8, 12, 20, 50, 100, 200])
    assert swish(x) == pytest.approx(
        x, abs=1e-1
    )
    assert swish.gradient(x) == pytest.approx(1, abs=1e-1)

    swish = activations.grab("swish", class_kwargs={"beta": other_beta})
    assert isinstance(swish, Swish)
    assert swish.beta == other_beta

    # x = 0 should still be 0
    assert swish(np.array([0])) == 0
    # x < 0 should still be negative and goes to 0
    assert swish(np.array([-1])) < 0

    # Swish is with higher beta less approximately the identity function, when x gets large
    # enough, thus gradient is 1.
    assert swish(np.array([4, 8, 12, 20, 50, 100, 200])) == pytest.approx(
        np.array([4, 8, 12, 20, 50, 100, 200]), abs=1e-10
    )
    assert swish.gradient(x) == pytest.approx(1, abs=1e-10)