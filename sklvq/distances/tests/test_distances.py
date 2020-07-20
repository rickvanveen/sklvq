import pytest
import numpy as np

from sklvq.misc.common_checks import has_call_method, has_gradient_method

# GLVQ Distances
from sklvq.distances import (
    Euclidean,
    SquaredEuclidean,
)

# GMLVQ Distances
from sklvq.distances import AdaptiveSquaredEuclidean

# LGMLVQ Distances
from sklvq.distances import (
    LocalAdaptiveSquaredEuclidean,
)


class LVQModel:
    """
    Fake class to test the distance functions as they for the distance function only really need
    the prototypes_ and some an omega_
    """

    def __init__(self, prototypes, omega):
        self.prototypes_ = prototypes
        self.omega_ = omega

    def get_model_params(self):
        if self.omega_ is None:
            return self.prototypes_
        else:
            return self.prototypes_, self.omega_


def _check_distance(distfun, data, model):

    assert has_call_method(distfun)

    assert has_gradient_method(distfun)

    dists = distfun(data, model)

    # Samples by prototypes
    assert np.all(dists.shape == (data.shape[0], model.prototypes_.shape[0]))

    # 'Never' negative
    assert np.all(distfun(data, model) >= 0.0)

    # a to b is equal to b to a (at the moment)
    assert dists[1, 0] == dists[0, 1]

    # Same values in negative direction is same as the samples in the positive direction.
    assert dists[0, 0] == dists[1, 1]

    for i_prototype in range(0, model.prototypes_.shape[1]):
        # Check if shape of the gradient makes sense
        gradient = distfun.gradient(data, model, 0)

        variables_size = (
            model.prototypes_.size + model.omega_.size
            if model.omega_ is not None
            else model.prototypes_.size
        )
        assert gradient.shape == (data.shape[0], variables_size)

        # Happens when data is exactly on the prototype for some distances.
        assert np.all(np.logical_not(np.isnan(gradient)))

        # This should also not happen
        assert np.all(np.logical_not(np.isinf(gradient)))


@pytest.mark.parametrize(
    "distance_class", [Euclidean, SquaredEuclidean],
)
def test_glvq_distance(distance_class):
    distfun = distance_class()
    # Some data and prototypes
    data = np.array([[1, 2, 3], [-1, -2, -3]])
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])

    model = LVQModel(p, None)

    _check_distance(distfun, data, model)

    # Still need to be able to deal with not-nan data.
    distfun = distance_class(force_all_finite="allow-nan")
    _check_distance(distfun, data, model)


@pytest.mark.parametrize(
    "distance_class", [Euclidean, SquaredEuclidean],
)
def test_glvq_nan_distance(distance_class):
    distfun = distance_class(force_all_finite="allow-nan")
    # Some data and prototypes
    data = np.array([[1, np.nan, 3], [-1, np.nan, -3]])
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])

    model = LVQModel(p, None)

    _check_distance(distfun, data, model)


@pytest.mark.parametrize(
    "distance_class", [AdaptiveSquaredEuclidean],
)
def test_gmlvq_distance(distance_class):
    distfun = distance_class()

    # Some data and prototypes
    data = np.array([[1, 2, 3], [-1, -2, -3]])
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])
    o = np.identity(data.shape[1])

    model = LVQModel(p, o)

    _check_distance(distfun, data, model)
    # Still need to be able to deal with not-nan data.
    distfun = distance_class(force_all_finite="allow-nan")
    _check_distance(distfun, data, model)

    # Rectangular omega
    o = o[:, 0:2].T
    model = LVQModel(p, o)

    distfun = distance_class()
    _check_distance(distfun, data, model)
    # Still need to be able to deal with not-nan data.
    distfun = distance_class(force_all_finite="allow-nan")
    _check_distance(distfun, data, model)


@pytest.mark.parametrize(
    "distance_class", [AdaptiveSquaredEuclidean],
)
def test_gmlvq_nan_distance(distance_class):
    distfun = distance_class(force_all_finite="allow-nan")

    # Some data and prototypes
    data = np.array([[1, np.nan, 3], [-1, np.nan, -3]])
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])
    o = np.identity(data.shape[1])

    model = LVQModel(p, o)

    _check_distance(distfun, data, model)

    # Rectangular omega
    o = o[:, 0:2].T
    model = LVQModel(p, o)

    _check_distance(distfun, data, model)

@pytest.mark.parametrize(
    "distance_class", [LocalAdaptiveSquaredEuclidean],
)
def test_lgmlvq_distance(distance_class):
    distfun = distance_class()

    # Some data and prototypes
    data = np.array([[1, 2, 3], [-1, -2, -3]])
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])

    shape = (data.shape[1], data.shape[1])
    o = np.array([np.eye(*shape) for _ in range(p.shape[0])])

    model = LVQModel(p, o)
    model.localization = "prototype"
    model.prototypes_labels_ = np.array([0, 1, 2])

    _check_distance(distfun, data, model)

    distfun = distance_class(force_all_finite="allow-nan")
    _check_distance(distfun, data, model)


@pytest.mark.parametrize(
    "distance_class", [LocalAdaptiveSquaredEuclidean],
)
def test_gmlvq_nan_distance(distance_class):
    distfun = distance_class(force_all_finite="allow-nan")

    # Some data and prototypes
    data = np.array([[1, np.nan, 3], [-1, np.nan, -3]])
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])
    o = np.identity(data.shape[1])

    shape = (data.shape[1], data.shape[1])
    o = np.array([np.eye(*shape) for _ in range(p.shape[0])])

    model = LVQModel(p, o)
    model.localization = "prototype"
    model.prototypes_labels_ = np.array([0, 1, 2])

    _check_distance(distfun, data, model)