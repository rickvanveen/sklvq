from abc import ABC, abstractmethod
import numpy as np


def _find_min(indices, distances):
    """ Helper function to find the minimum distance and the index of this distance. """
    dist_temp = np.where(indices, distances, np.inf)
    return dist_temp.min(axis=1), dist_temp.argmin(axis=1)


def _relative_distance(dist_same, dist_diff):
    return (dist_same - dist_diff) / (dist_same + dist_diff)


def _relative_distance_grad(dist_same, dist_diff):
    return 2 * dist_diff / (dist_same + dist_diff) ** 2


class AbstractObjective(ABC):

    def __init__(self, distance):
        self.distance = distance

    @abstractmethod
    def __call__(self, variables, prototype_labels, data, labels):
        raise NotImplementedError("You should implement this! Should return"
                                  " objective value, gradient (1D, like variables)")


class RelativeDistanceObjective(AbstractObjective):
    def __init__(self, distance=None, scaling=None, prototypes_shape=None):
        self.prototypes_shape = prototypes_shape
        self.scaling = scaling
        super(RelativeDistanceObjective, self).__init__(distance)

    # Computes all the distances such that we only need to do this once.
    def _compute_distance(self, data, labels, prototypes, prototype_labels):
        """ Computes the distances between each prototype and each observation and finds all indices where the shortest
        distance is that of the prototype with the same label and with a different label. """

        distances = self.distance(data, prototypes)

        ii_same = np.transpose(np.array([labels == prototype_label for prototype_label in prototype_labels]))
        ii_diff = ~ii_same

        dist_same, i_dist_same = _find_min(ii_same, distances)
        dist_diff, i_dist_diff = _find_min(ii_diff, distances)

        return dist_same, dist_diff, i_dist_same, i_dist_diff

    def _gradient_scaling(self, relative_distance, dist_same, dist_diff):
        gradient_scaling = (  # All samples where the current prototype is the closest and has the same label
                self.scaling.gradient(relative_distance) *
                _relative_distance_grad(dist_same, dist_diff))
        return np.atleast_2d(gradient_scaling)

    # (dS/dmu * dmu/dd_i * dd_i/dw_i) for i J and K
    def _prototype_gradient(self, gradient_scaling, data, prototype):
        # d'_J(x, w) for all x (samples) in data and the current prototype
        dist_grad_wrt_prototype = np.atleast_2d(self.distance.gradient(data, prototype))
        return gradient_scaling.dot(dist_grad_wrt_prototype)

    # dS/dw_J() = (dS/dmu * dmu/dd_J * dd_J/dw_J) - (dS/dmu * dmu/dd_K * dd_K/dw_K)
    def _gradient(self, dist_same, dist_diff, i_dist_same, i_dist_diff,
                  relative_distance, data, prototypes, prototypes_labels):
        gradient = np.zeros(self.prototypes_shape)
        for i_prototype in range(0, prototypes_labels.size):
            # Find for which samples this prototype is the closest and has the same label
            ii_same = i_prototype == i_dist_same
            if any(ii_same):
                gradient_scaling = self._gradient_scaling(relative_distance[ii_same], dist_same[ii_same],
                                                          dist_diff[ii_same])
                gradient[i_prototype] = (
                        gradient[i_prototype] + self._prototype_gradient(gradient_scaling, data[ii_same, :],
                                                                         prototypes[i_prototype, :]))

            # Find for which samples this prototype is the closest and has a different label
            ii_diff = i_prototype == i_dist_diff
            if any(ii_diff):
                gradient_scaling = self._gradient_scaling(relative_distance[ii_diff], dist_diff[ii_diff],
                                                          dist_same[ii_diff])
                gradient[i_prototype] = (
                        gradient[i_prototype] - self._prototype_gradient(gradient_scaling, data[ii_diff, :],
                                                                         prototypes[i_prototype, :]))
        return gradient.ravel()

    # S() = sum(f(mu(x)))
    def _cost(self, relative_distance):
        return np.sum(self.scaling(relative_distance))

    def restore_from_variables(self, variables):
        return variables.reshape(self.prototypes_shape)

    def __call__(self, variables, prototypes_labels, data, labels):
        # Variables 1D -> 2D prototypes
        prototypes = self.restore_from_variables(variables)

        # dist_same, d_J(X, w_J), contains the distances between all
        # samples X and the closest prototype with the same label (_J)
        # i_dist_same tells you which prototype was closest.

        # dist_diff, d_K(X, w_K), contains the distances between all
        # samples X and the closest prototype with a different label (_K)
        # i_dist_diff same size as dist_diff, contains indices corresponding to the label of the closest prototype
        dist_same, dist_diff, i_dist_same, i_dist_diff = self._compute_distance(data, labels, prototypes,
                                                                                prototypes_labels)
        # mu(x)
        relative_distance = _relative_distance(dist_same, dist_diff)

        # First part is S = Sum(f(u(x))), Second part: gradient of all the prototypes "flattened" (ravel()) -> 1D array
        return self._cost(relative_distance), self._gradient(dist_same, dist_diff, i_dist_same,
                                                             i_dist_diff, relative_distance,
                                                             data, prototypes, prototypes_labels)


class RelevanceRelativeDistanceObjective(RelativeDistanceObjective):

    def __init__(self, distance=None, scaling=None, prototypes_shape=None, omega_shape=None):
        self.omega_shape = omega_shape
        super(RelevanceRelativeDistanceObjective, self).__init__(distance, scaling, prototypes_shape)

    # TODO: Omega data structure omega.normalise() makes so much more sense...
    def _normalise(self, omega):
        return omega / np.sqrt(np.sum(np.diagonal(omega.T.dot(omega))))

    def restore_from_variables(self, variables):
        prototypes_size = np.prod(self.prototypes_shape)
        prototypes_variables = variables[:prototypes_size]
        prototypes = prototypes_variables.reshape(self.prototypes_shape)

        # TODO: Works for Global but not Local matrices...
        omega_variables = variables[prototypes_size:]
        omega = omega_variables.reshape(self.omega_shape)

        return prototypes, self._normalise(omega)

    def _omega_gradient(self, gradient_scaling, data, prototype):
        dist_grad_wrt_omega = np.atleast_2d(self.distance.omega_gradient(data, prototype))
        return gradient_scaling.dot(dist_grad_wrt_omega)

    def _gradient(self, dist_same, dist_diff, i_dist_same, i_dist_diff,
                  relative_distance, data, prototypes, prototypes_labels):
        gradient_prototypes = np.zeros(self.prototypes_shape)
        gradient_omega = np.zeros(self.omega_shape).ravel() # Already 1D because easier...

        for i_prototype in range(0, prototypes_labels.size):
            ii_same = i_prototype == i_dist_same
            if any(ii_same):
                gradient_scaling = self._gradient_scaling(relative_distance[ii_same], dist_same[ii_same],
                                                          dist_diff[ii_same])
                gradient_prototypes[i_prototype] = (
                        gradient_prototypes[i_prototype] + self._prototype_gradient(gradient_scaling, data[ii_same, :],
                                                                                    prototypes[i_prototype, :]))
                gradient_omega = gradient_omega + self._omega_gradient(gradient_scaling, data[ii_same, :],
                                                                       prototypes[i_prototype, :])

            # Find for which samples this prototype is the closest and has a different label
            ii_diff = i_prototype == i_dist_diff
            if any(ii_diff):
                gradient_scaling = self._gradient_scaling(relative_distance[ii_diff], dist_diff[ii_diff],
                                                      dist_same[ii_diff])
                gradient_prototypes[i_prototype] = (
                    gradient_prototypes[i_prototype] - self._prototype_gradient(gradient_scaling, data[ii_diff, :],
                                                                            prototypes[i_prototype, :]))
                gradient_omega = (gradient_omega - self._omega_gradient(gradient_scaling, data[ii_diff, :],
                                                                        prototypes[i_prototype, :]))
        return np.append(gradient_prototypes.ravel(), gradient_omega.ravel())

    def __call__(self, variables, prototypes_labels, data, labels):
        prototypes, omega = self.restore_from_variables(variables)

        self.distance.omega = omega
        dist_same, dist_diff, i_dist_same, i_dist_diff = self._compute_distance(data, labels,
                                                                                prototypes, prototypes_labels)
        relative_distance = _relative_distance(dist_same, dist_diff)

        return self._cost(relative_distance), self._gradient(dist_same, dist_diff, i_dist_same, i_dist_diff,
                                                             relative_distance, data, prototypes, prototypes_labels)
