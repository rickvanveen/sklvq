import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sklvq.distances import LocalAdaptiveSquaredEuclidean
from tests.distance_cases import make_distance_model
from tests.distances._utils import finite_difference


@pytest.fixture
def class_model():
    # Two prototypes share each class factor, making class and prototype indices distinguishable.
    return make_distance_model(
        [[0, 0], [3, 0], [0, 4], [3, 4]],
        [[[1, 0], [0, 2]], [[2, 0], [0, 1]]],
        localization="class",
    )


@pytest.fixture
def prototype_model():
    # Same-class prototypes deliberately have different factors.
    return make_distance_model(
        [[0, 0], [3, 0], [0, 4], [3, 4]],
        [[[1, 0], [0, 2]], [[2, 0], [0, 1]], [[3, 0], [0, 1]], [[1, 0], [0, 1]]],
        localization="prototypes",
    )


def test_class_local_distances(class_model):
    # Prototypes 0/1 double y; prototypes 2/3 double x.
    # For sample [3,4], the projections [3,8], [0,8], [6,0], [0,0] give 73,64,36,0.
    expected = np.array([[73, 64, 36, 0], [0, 9, 16, 52]])
    actual = LocalAdaptiveSquaredEuclidean()(np.array([[3, 4], [0, 0]], dtype=float), class_model)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected)


def test_prototype_local_distances(prototype_model):
    # Each prototype has its own factor, even when two prototypes share a class.
    # For sample [3,4], the projections [3,8], [0,4], [9,0], [0,0] give 73,16,81,0.
    expected = np.array([[73, 16, 81, 0], [0, 36, 16, 25]])
    actual = LocalAdaptiveSquaredEuclidean()(np.array([[3, 4], [0, 0]], dtype=float), prototype_model)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected)


@pytest.mark.parametrize("localization", ["class", "prototypes"])
def test_rectangular_relevance(localization):
    """Check a discarded projection direction and one-sample batches handled by local slicing/squeezing."""
    factors = [[[1, -1]]] * (2 if localization == "class" else 4)
    model = make_distance_model([[0, 0], [3, 0], [0, 4], [3, 4]], factors, localization)

    # Projection onto a-b discards the difference between [3,3] and the origin.
    # Projected differences are [-1,-4,3,0] and [0,-3,4,1]; distances are their squares.
    expected = np.array([[1, 16, 9, 0], [0, 9, 16, 1]])
    actual = LocalAdaptiveSquaredEuclidean()(np.array([[3, 4], [3, 3]], dtype=float), model)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected)

    # Stochastic training evaluates one sample against multiple prototypes, not a one-prototype model.
    expected_single = expected[:1]
    actual_single = LocalAdaptiveSquaredEuclidean()(np.array([[3.0, 4.0]]), model)
    assert actual_single.shape == expected_single.shape
    assert_allclose(actual_single, expected_single)


@pytest.mark.parametrize("localization", ["class", "prototypes"])
@pytest.mark.parametrize("missing", [False, True], ids=["finite", "nan"])
def test_input_preservation(localization, missing):
    """Keep caller-owned samples, prototypes and local matrices intact after finite or NaN evaluations."""
    distance = LocalAdaptiveSquaredEuclidean()
    data = np.array([[3.0, np.nan if missing else 4.0]])
    model = make_distance_model([[0, 0], [1, 0]], [[[1, 0], [0, 1]], [[2, 0], [0, 1]]], localization)
    model.force_all_finite = "allow-nan" if missing else True
    originals = data.copy(), model.prototypes_.copy(), model.omega_.copy()

    # Check after each call so an expected numerical defect cannot conceal a mutation.
    distance(data, model)
    for actual, original in zip((data, model.prototypes_, model.omega_), originals):
        assert_array_equal(actual, original)

    for i in range(2):
        distance.gradient(data, model, i)
        for actual, original in zip((data, model.prototypes_, model.omega_), originals):
            assert_array_equal(actual, original)


def test_class_local_gradient(class_model):
    """Prototype 3 belongs to class 1: finite differences expose selection by prototype rather than class."""
    distance = LocalAdaptiveSquaredEuclidean()
    data = np.array([[1, 2], [-2, 1]], dtype=float)

    # Perturb the shared class factor, not the factor at the prototype's index.
    parameters = [class_model.prototypes_[3], class_model.omega_[1]]
    expected_gradient = finite_difference(lambda: distance(data, class_model)[:, 3], parameters)
    actual_gradient = distance.gradient(data, class_model, 3)

    assert actual_gradient.shape == expected_gradient.shape
    assert_allclose(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("i_prototype", [0, 3])
def test_prototype_local_gradient(prototype_model, i_prototype):
    """Check each selected factor's derivatives; prototype 3 distinguishes prototype from class indexing."""
    distance = LocalAdaptiveSquaredEuclidean()
    data = np.array([[1, 2], [-2, 1]], dtype=float)

    # Prototype coordinates come first, followed by its own factor entries in row-major order.
    parameters = [prototype_model.prototypes_[i_prototype], prototype_model.omega_[i_prototype]]
    expected_gradient = finite_difference(lambda: distance(data, prototype_model)[:, i_prototype], parameters)
    actual_gradient = distance.gradient(data, prototype_model, i_prototype)

    assert actual_gradient.shape == expected_gradient.shape
    assert_allclose(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-7)


@pytest.fixture(params=["class", "prototypes"])
def missing_model(request):
    # Off-diagonal relevance exposes missing-coordinate derivatives in both localization modes.
    model = make_distance_model([[0, 0, 0]], [[[1, 1, 0], [0, 1, 1]]], request.param)
    model.force_all_finite = "allow-nan"
    return model


def test_missing_distance(missing_model):
    # Masked difference [3,-4,0] projects to [-1,-4], with squared length 17, not 289.
    expected = [[17]]
    actual = LocalAdaptiveSquaredEuclidean()(np.array([[3, -4, np.nan]]), missing_model)
    assert_allclose(actual, expected)


def test_missing_gradient(missing_model):
    """Check fixed masked derivatives, including the zero update for an entirely missing record."""
    distance = LocalAdaptiveSquaredEuclidean()
    data = np.array([[3, -4, np.nan], [np.nan, np.nan, np.nan]])

    # Keep valid observed-coordinate derivatives outside the expected-failure regression.
    # Columns: prototype x,y,z; then omega row 0 and row 1. The all-missing sample supplies only zeros.
    # Observed prototype entries come from -2*Omega.T@[-1,-4]; omega uses 2*outer([-1,-4], [3,-4,0]).
    expected_gradient = np.array([[2, 10, 0, -6, 8, 0, -24, 32, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    actual_gradient = distance.gradient(data, missing_model, 0)
    assert actual_gradient.shape == expected_gradient.shape
    assert_allclose(actual_gradient[:, :2], expected_gradient[:, :2])
    assert_allclose(actual_gradient[:, 3:], expected_gradient[:, 3:])

    # Check all-missing behavior outside the partial-NaN expected failures.
    expected_all_missing_distance = [[0]]
    actual_all_missing_distance = distance(data[1:], missing_model)
    assert_array_equal(actual_all_missing_distance, expected_all_missing_distance)
    assert_array_equal(actual_gradient[1], expected_gradient[1])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Local adaptive gradient updates a missing coordinate")
def test_missing_coordinate_gradient(missing_model):
    # Masking removes dependence on the last prototype coordinate, regardless of local relevance.
    expected_missing_derivative = [0]
    actual = LocalAdaptiveSquaredEuclidean().gradient(np.array([[3, -4, np.nan]]), missing_model, 0)
    assert_array_equal(actual[:, 2], expected_missing_derivative)
