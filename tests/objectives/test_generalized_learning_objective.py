import numpy as np
import pytest

from sklvq import GLVQ
from sklvq.discriminants import RelativeDistance
from sklvq.distances import SquaredEuclidean
from sklvq.objectives import GeneralizedLearningObjective
from tests.distances._utils import finite_difference


@pytest.fixture
def model():
    # Use GLVQ's real gradient layout without fitting or moving the prototypes.
    model = GLVQ()
    model._variables = np.array([0.0, 8.0, 4.0, 20.0])
    model.prototypes_ = model.get_variables().reshape(-1, 1)
    model._prototypes_shape = model.prototypes_.shape
    model.n_features_in_ = 1
    model.prototypes_labels_ = np.array([0, 0, 1, 1])
    model._distance = SquaredEuclidean()
    return model


@pytest.fixture
def objective():
    # Scores of ±0.8 then have sigmoid odds of 3:1 or 1:3.
    return GeneralizedLearningObjective("sigmoid", {"beta": np.log(3) / 0.8}, "relative-distance", None)


def test_cost(objective, model):
    """Sum each sample's activated score using the nearest prototype on each side."""
    data = np.array([[1.0], [5.0], [7.0], [2.0]])
    labels = np.array([0, 0, 1, 1])
    original_data, original_labels, original_variables = data.copy(), labels.copy(), model.get_variables().copy()

    # Winning squared distances (same, different): (1,9), (9,1), (9,1), (4,4).
    expected = 0.25 + 0.75 + 0.75 + 0.5
    actual = objective(model, data, labels)
    assert np.ndim(actual) == 0
    assert actual == pytest.approx(expected)
    np.testing.assert_array_equal(model.get_variables(), original_variables)
    np.testing.assert_array_equal(data, original_data)
    np.testing.assert_array_equal(labels, original_labels)


@pytest.mark.parametrize("batch_size", [1, 4], ids=["single-sample", "full-batch"])
def test_gradient(objective, model, batch_size):
    """Check solver-order derivatives, including repeated winners and an unused prototype."""
    data = np.array([[1.0], [5.0], [7.0], [2.0]])[:batch_size]
    labels = np.array([0, 0, 1, 1])[:batch_size]
    variables = model.get_variables()
    expected = finite_difference(lambda: objective(model, data, labels), [variables])[0]

    original_data, original_labels, original_variables = data.copy(), labels.copy(), variables.copy()
    actual = objective.gradient(model, data, labels)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(variables, original_variables)
    np.testing.assert_array_equal(data, original_data)
    np.testing.assert_array_equal(labels, original_labels)


def test_keyword_only_discriminant_gradient(model):
    """A custom discriminant can use the keyword-only label flag declared by its base class."""

    class KeywordOnlyDistance(RelativeDistance):
        def gradient(self, dist_same, dist_diff, *, same_label):
            return super().gradient(dist_same, dist_diff, same_label=same_label)

    objective = GeneralizedLearningObjective("identity", None, KeywordOnlyDistance, None)
    # At x=1 the winning prototypes are 0 and 4; the others receive no gradient.
    actual = objective.gradient(model, np.array([[1.0]]), np.array([0]))
    np.testing.assert_allclose(actual, [-0.36, 0.0, -0.12, 0.0])
