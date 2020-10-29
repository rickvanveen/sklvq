import numpy as np

from sklvq import distances
from sklvq._utils import init_class
from sklvq.distances import ALIASES
from sklvq.distances import DistanceBaseClass


class DummyLVQ:
    """
    Fake class to test the distance functions as they for the distance function only really need
    the prototypes_ and some an omega_
    """

    def __init__(self, prototypes, labels=None, omega=None):
        self._prototypes_shape = prototypes.shape
        self._prototypes_size = prototypes.size

        self.prototypes_labels_ = labels

        variables_size = prototypes.size

        if omega is not None:
            self._omega_shape = omega.shape
            self._omega_size = omega.size
            variables_size += self._omega_size

        self._variables = np.empty(variables_size)

        self.prototypes_ = self.to_prototypes(self._variables)
        np.copyto(self.prototypes_, prototypes)

        if omega is not None:
            self.omega_ = self.to_omega(self._variables)
            np.copyto(self.omega_, omega)
        else:
            self.omega_ = None

    def to_prototypes(self, var_buffer):
        return var_buffer[: self._prototypes_size].reshape(self._prototypes_shape)

    def to_omega(self, var_buffer):
        return var_buffer[self._prototypes_size :].reshape(self._omega_shape)

    def get_model_params(self):
        if self.omega_ is not None:
            return self.prototypes_, self.omega_
        else:
            return self.prototypes_

    @staticmethod
    def _compute_lambda(omega):
        return np.einsum("ji, jk -> ik", omega, omega)


def check_init_distance(distance_string):
    distatance_class = init_class(distances, distance_string)

    assert isinstance(distatance_class, type)

    distance_inistance = distatance_class()

    assert isinstance(distance_inistance, DistanceBaseClass)

    return distatance_class


def test_aliases():
    for value in ALIASES.values():
        check_init_distance(value)


def check_distance(distfun, data, model):

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

        gradient_size = model.prototypes_[0, :].size
        if model.omega_ is not None:
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