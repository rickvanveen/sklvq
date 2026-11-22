import numpy as np

from sklvq.activations import Swish


def test_swish():
    """At beta=log(9), sigmoid weights are 0.1, 0.5 and 0.9; the left slope is negative."""
    beta = np.log(9)
    activation = Swish(beta=beta)
    scores = np.array([-1.0, 0.0, 1.0])
    expected = [-0.1, 0.0, 0.9]
    # Product rule: the slope adds x times the sigmoid slope to each weight.
    expected_gradient = [0.1 - beta * 0.09, 0.5, 0.9 + beta * 0.09]

    actual = activation(scores)
    actual_gradient = activation.gradient(scores)
    assert actual.shape == actual_gradient.shape == scores.shape
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual_gradient, expected_gradient)
