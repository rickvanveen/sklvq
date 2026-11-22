import numpy as np

from sklvq.activations import Sigmoid


def test_sigmoid():
    """At scores −1, 0 and 1, beta=log(3) gives odds of 1:3, 1:1 and 3:1."""
    beta = np.log(3)
    activation = Sigmoid(beta=beta)
    scores = np.array([-1.0, 0.0, 1.0])
    expected = [0.25, 0.5, 0.75]
    expected_gradient = beta * np.array([3 / 16, 1 / 4, 3 / 16])

    actual = activation(scores)
    actual_gradient = activation.gradient(scores)
    assert actual.shape == actual_gradient.shape == scores.shape
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual_gradient, expected_gradient)


def test_saturated_sigmoid_gradient():
    """A saturated sigmoid should give the optimizer zero weight, not NaN."""
    actual = Sigmoid(beta=1000).gradient(np.array([1.0]))
    np.testing.assert_array_equal(actual, [0.0])
