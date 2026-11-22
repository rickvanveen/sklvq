import numpy as np

from sklvq.activations import SoftPlus


def test_soft_plus():
    """At beta=log(3), the exponentials are 1/3, 1 and 3; the slope includes beta."""
    beta = np.log(3)
    activation = SoftPlus(beta=beta)
    scores = np.array([-1.0, 0.0, 1.0])
    expected = np.log([4 / 3, 2, 4])
    expected_gradient = beta * np.array([0.25, 0.5, 0.75])

    actual = activation(scores)
    actual_gradient = activation.gradient(scores)
    assert actual.shape == actual_gradient.shape == scores.shape
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual_gradient, expected_gradient)


def test_positive_tail():
    """At beta*x=1000 the loss is indistinguishable from 1000 in double precision."""
    actual = SoftPlus(beta=1000)(np.array([1.0]))
    np.testing.assert_allclose(actual, [1000.0])


def test_positive_tail_gradient():
    """The slope in the positive tail should approach beta rather than become NaN."""
    actual = SoftPlus(beta=1000).gradient(np.array([1.0]))
    np.testing.assert_allclose(actual, [1000.0])


def test_negative_tail():
    """The tiny loss at beta*x=-40 is still representable and must not round to zero."""
    actual = SoftPlus(beta=40)(np.array([-1.0]))
    np.testing.assert_allclose(actual, [np.exp(-40)], atol=0)
