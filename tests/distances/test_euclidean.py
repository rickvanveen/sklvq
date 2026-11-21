import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sklvq.distances import Euclidean, SquaredEuclidean
from tests.distance_cases import make_distance_model
from tests.distances._utils import finite_difference

pytestmark = pytest.mark.parametrize(
    ("distance_type", "power"), [(Euclidean, 1), (SquaredEuclidean, 2)], ids=["euclidean", "squared"]
)


def test_known_distances(distance_type, power):
    # A 3-by-4 rectangle has side lengths 3 and 4 and diagonal length 5.
    data = np.array([[3, 4], [0, 0]], dtype=float)
    model = make_distance_model([[0, 0], [3, 0], [0, 4], [3, 4]])

    # Rows follow samples; columns follow prototypes. Squared Euclidean returns squared side/diagonal lengths.
    expected = np.array([[5, 4, 3, 0], [0, 3, 4, 5]] if power == 1 else [[25, 16, 9, 0], [0, 9, 16, 25]])
    actual = distance_type()(data, model)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected)


@pytest.mark.parametrize("missing", [False, True], ids=["finite", "nan"])
def test_input_preservation(distance_type, power, missing):
    """Training reuses these arrays; neither evaluation nor NaN masking may overwrite them."""
    distance = distance_type()
    data = np.array([[3.0, np.nan if missing else 4.0]])
    model = make_distance_model([[0, 0], [3, 0]])
    model.force_all_finite = "allow-nan" if missing else True
    originals = data.copy(), model.prototypes_.copy()

    # Check each call separately so later calls cannot hide an earlier modification.
    distance(data, model)
    for actual, original in zip((data, model.prototypes_), originals):
        assert_array_equal(actual, original)

    for i in range(2):
        distance.gradient(data, model, i)
        for actual, original in zip((data, model.prototypes_), originals):
            assert_array_equal(actual, original)


def test_gradient(distance_type, power):
    """Check the selected prototype's derivative numerically, and the zero convention at coincidence."""
    distance = distance_type()
    data = np.array([[1, 2], [-2, 1]], dtype=float)
    model = make_distance_model([[0, 0], [3, 0]])

    # A nonzero index catches accidental use of the first prototype without repeating equivalent indices.
    expected_gradient = finite_difference(lambda: distance(data, model)[:, 1], [model.prototypes_[1]])
    actual_gradient = distance.gradient(data, model, 1)

    assert actual_gradient.shape == expected_gradient.shape
    assert_allclose(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-7)

    # Euclidean chooses a zero subgradient at nondifferentiable coincidence.
    expected_at_coincidence = [[0, 0]]
    actual_at_coincidence = distance.gradient(model.prototypes_[1:2], model, 1)
    assert_array_equal(actual_at_coincidence, expected_at_coincidence)


def test_missing_distance(distance_type, power):
    # Opposite signs expose square-of-sum errors; masking leaves a 3-4-5 triangle.
    model = make_distance_model([[0, 0, 0]])
    model.force_all_finite = "allow-nan"

    expected = [[5]] if power == 1 else [[25]]
    actual = distance_type()(np.array([[3, -4, np.nan]]), model)
    assert_allclose(actual, expected)


def test_missing_gradient(distance_type, power):
    """Check fixed derivatives for partial and entirely missing records, including zero-denominator handling."""
    model = make_distance_model([[0, 0, 0]])
    model.force_all_finite = "allow-nan"
    distance = distance_type()
    data = np.array([[3, -4, np.nan], [np.nan, np.nan, np.nan]])

    # The partial observation has a 3-4-5 direction; the empty observation supplies no update.
    # Prototype derivatives are -difference/5 for Euclidean or -2*difference for squared Euclidean.
    expected = [[-0.6, 0.8, 0], [0, 0, 0]] if power == 1 else [[-6, 8, 0], [0, 0, 0]]
    expected = np.array(expected)
    actual = distance.gradient(data, model, 0)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected)

    # With nothing observed, the distance is zero too.
    expected_all_missing = [[0]]
    actual_all_missing = distance(data[1:], model)
    assert_array_equal(actual_all_missing, expected_all_missing)
