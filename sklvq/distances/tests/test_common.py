import numpy as np

from sklvq import distances
from sklvq._utils import init_class
from sklvq.distances import DistanceBaseClass
from sklvq.models import GMLVQ, LGMLVQ


def check_init_distance(distance_string):
    distatance_class = init_class(distances, distance_string)

    assert isinstance(distatance_class, type)

    distance_instance = distatance_class()

    assert isinstance(distance_instance, DistanceBaseClass)


# When there are distance functions with aliases re-enable
# def test_aliases():
#     for value in ALIASES.keys():
#         check_init_distance(value)


def check_distance(distfun, data, model):

    dists = distfun(data, model)
    # Samples by prototypes
    assert np.all(dists.shape == (data.shape[0], model.prototypes_.shape[0]))

    # 'Never' negative
    assert np.all(distfun(data, model) >= 0.0)

    # a to b is equal to b to a
    assert dists[1, 0] == dists[0, 1]

    # Same values in negative direction is same as the samples in the positive direction.
    assert dists[0, 0] == dists[1, 1]

    for i_prototype in range(0, model.prototypes_.shape[1]):
        # Check if shape of the gradient makes sense
        gradient = distfun.gradient(data, model, 0)

        gradient_size = model.prototypes_[0, :].size

        if isinstance(model, (GMLVQ, LGMLVQ)):
            if model.omega_.ndim == 3:
                gradient_size += model.omega_[0, :, :].size
            else:
                gradient_size += model.omega_.size

        # Size of num_samples x single prototype...
        assert gradient.shape == (data.shape[0], gradient_size)

        # Happens when X is exactly on the prototype for some distances.
        assert np.all(np.logical_not(np.isnan(gradient)))

        # This should also not happen
        assert np.all(np.logical_not(np.isinf(gradient)))
