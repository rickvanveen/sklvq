import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sklvq.distances import AdaptiveSquaredEuclidean
from tests.distance_cases import make_distance_model
from tests.distances._utils import finite_difference


def test_known_distances():
    # Omega maps [a,b] to [a+b,b]; [3,4] therefore has squared length 65.
    data = np.array([[3, 4], [0, 0]], dtype=float)
    model = make_distance_model([[0, 0], [3, 0], [0, 4], [3, 4]], [[1, 1], [0, 1]])

    # For [3,4], projected differences are [7,4], [4,4], [3,0], [0,0]: squared lengths 65,32,9,0.
    expected = np.array([[65, 32, 9, 0], [0, 9, 32, 65]])
    actual = AdaptiveSquaredEuclidean()(data, model)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_rectangular_relevance():
    # Projection onto a-b gives zero distance even for distinct points [3,3] and [0,0].
    model = make_distance_model([[0, 0], [3, 0], [0, 4], [3, 4]], [[1, -1]])

    # Projected differences are [-1,-4,3,0] and [0,-3,4,1]; distances are their squares.
    expected = np.array([[1, 16, 9, 0], [0, 9, 16, 1]])
    actual = AdaptiveSquaredEuclidean()(np.array([[3, 4], [3, 3]], dtype=float), model)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected)


@pytest.mark.parametrize("missing", [False, True], ids=["finite", "nan"])
def test_input_preservation(missing):
    """Check that distance and gradient calls preserve shared data and parameters, including during masking."""
    distance = AdaptiveSquaredEuclidean()
    data = np.array([[3.0, np.nan if missing else 4.0]])
    model = make_distance_model([[0, 0], [3, 0]], [[1, 1], [0, 1]])
    model.force_all_finite = "allow-nan" if missing else True
    originals = data.copy(), model.prototypes_.copy(), model.omega_.copy()

    # Check after each operation, independently of numerical correctness.
    distance(data, model)
    for actual, original in zip((data, model.prototypes_, model.omega_), originals):
        assert_array_equal(actual, original)

    for i in range(2):
        distance.gradient(data, model, i)
        for actual, original in zip((data, model.prototypes_, model.omega_), originals):
            assert_array_equal(actual, original)


def test_gradient():
    """Check prototype and rectangular-factor derivatives in the order consumed by model updates."""
    distance = AdaptiveSquaredEuclidean()
    data = np.array([[1, 2], [-2, 1]], dtype=float)
    model = make_distance_model([[0, 0], [3, 0]], [[1, -1]])

    # Use a nonzero prototype index and perturb raw parameters without normalization.
    parameters = [model.prototypes_[1], model.omega_]
    expected_gradient = finite_difference(lambda: distance(data, model)[:, 1], parameters)
    actual_gradient = distance.gradient(data, model, 1)

    assert actual_gradient.shape == expected_gradient.shape
    assert_allclose(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-7)


@pytest.fixture
def missing_model():
    # Off-diagonal relevance lets an observed feature influence another prototype coordinate.
    model = make_distance_model([[0, 0, 0]], [[1, 1, 0], [0, 1, 1]])
    model.force_all_finite = "allow-nan"
    return model


def test_missing_distance(missing_model):
    # Masked difference [3,-4,0] projects to [-1,-4], with squared length 17.
    expected = [[17]]
    actual = AdaptiveSquaredEuclidean()(np.array([[3, -4, np.nan]]), missing_model)
    assert_allclose(actual, expected)


def test_missing_gradient(missing_model):
    """Check fixed masked derivatives and zero contribution from a wholly missing record."""
    distance = AdaptiveSquaredEuclidean()
    data = np.array([[3, -4, np.nan], [np.nan, np.nan, np.nan]])

    # Columns: prototype x,y,z; then omega row 0 and row 1. The all-missing sample supplies only zeros.
    # Observed prototype entries come from -2*Omega.T@[-1,-4]; omega uses 2*outer([-1,-4], [3,-4,0]).
    expected_gradient = np.array([[2, 10, 0, -6, 8, 0, -24, 32, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    actual_gradient = distance.gradient(data, missing_model, 0)
    assert actual_gradient.shape == expected_gradient.shape
    assert_allclose(actual_gradient, expected_gradient)

    # With nothing observed, the distance and every derivative must be zero.
    expected_all_missing_distance = [[0]]
    actual_all_missing_distance = distance(data[1:], missing_model)
    assert_array_equal(actual_all_missing_distance, expected_all_missing_distance)
    assert_array_equal(actual_gradient[1], expected_gradient[1])
