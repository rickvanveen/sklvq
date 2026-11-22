import numpy as np
import pytest

from sklvq import discriminants
from sklvq._utils import init_class
from sklvq.discriminants import RelativeDistance


def test_relative_distance_name():
    """The model's configuration string selects the built-in relative distance."""
    assert init_class(discriminants, "relative-distance") is RelativeDistance


def test_relative_distance():
    """Check normalized scores for closer, tied and exactly matching prototypes."""
    dist_same = np.array([1.0, 3.0, 2.0, 0.0, 4.0])
    dist_diff = np.array([3.0, 1.0, 2.0, 4.0, 0.0])
    expected = np.array([-0.5, 0.5, 0.0, -1.0, 1.0])

    # Training reuses these distances to compute both gradient weights.
    original_same, original_diff = dist_same.copy(), dist_diff.copy()
    actual = RelativeDistance()(dist_same, dist_diff)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(dist_same, original_same)
    np.testing.assert_array_equal(dist_diff, original_diff)


@pytest.mark.parametrize("same_label", [True, False], ids=["same-label", "different-label"])
def test_gradient(same_label):
    """Perturb only the winning distance to check the weight used in the objective gradient."""
    discriminant = RelativeDistance()
    dist_same = np.array([1.0, 4.0])
    dist_diff = np.array([3.0, 2.0])
    step = 1e-5

    if same_label:
        plus = discriminant(dist_same + step, dist_diff)
        minus = discriminant(dist_same - step, dist_diff)
    else:
        plus = discriminant(dist_same, dist_diff + step)
        minus = discriminant(dist_same, dist_diff - step)
    expected_gradient = (plus - minus) / (2 * step)

    actual_gradient = discriminant.gradient(dist_same, dist_diff, same_label)

    assert actual_gradient.shape == expected_gradient.shape
    np.testing.assert_allclose(actual_gradient, expected_gradient)


@pytest.mark.parametrize(("same_label", "factor"), [(True, 3 / 8), (False, -1 / 8)])
def test_gradient_at_extreme_scales(same_label, factor):
    """At distances (s, 3s), the slopes are 6s/(4s)² and -2s/(4s)², both finite."""
    scales = np.array([1e-200, 1e200])
    expected = factor / scales
    actual = RelativeDistance().gradient(scales, 3 * scales, same_label)

    np.testing.assert_allclose(actual, expected, atol=0)
