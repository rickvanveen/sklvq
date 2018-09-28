from abc import ABC, abstractmethod
import numpy as np


def _find_min(indices, distances):
    """ Helper function to find the minimum distance and the index of this distance. """
    dist_temp = np.where(indices, distances, np.inf)
    return dist_temp.min(axis=1), dist_temp.argmin(axis=1)


# mu function... TODO: mu now not configurable like the rest, but could be?
def _relative_distance(dist_same, dist_diff):
    return (dist_same - dist_diff) / (dist_same + dist_diff)


def _relative_distance_grad(dist_same, dist_diff):
    return 2 * dist_diff / (dist_same + dist_diff) ** 2


# Strategy
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

    # (dS/dmu * dmu/dd_i * dd_i/dw_i) for i J and K
    def _partial_gradient(self, ii_samples, relative_distance, dist_same, dist_diff, data, prototype):
        relative_dist = (  # All samples where the current prototype is the closest and has the same label
                self.scaling.gradient(relative_distance[ii_samples]) *
                _relative_distance_grad(dist_same[ii_samples], dist_diff[ii_samples]))

        # d'_J(x, w) for all x (samples) in data and the current prototype
        dist_grad = self.distance.gradient(data[ii_samples, :], prototype)

        return relative_dist.dot(dist_grad)

    # TODO: Handle the case where there is no data... the current prototype is never the closest with different/same label.
    # dS/dw_J() = (dS/dmu * dmu/dd_J * dd_J/dw_J) - (dS/dmu * dmu/dd_K * dd_K/dw_K)
    def _gradient(self, dist_same, dist_diff, i_dist_same, i_dist_diff,
                  relative_distance, data, prototypes, prototypes_labels):

        gradient = np.zeros(self.prototypes_shape)
        for i_prototype in range(0, prototypes_labels.size):
            # Find for which samples this prototype is the closest and has the same label
            ii_same = i_prototype == i_dist_same
            gradient[i_prototype] = self._partial_gradient(ii_same, relative_distance, dist_same,
                                                           dist_diff, data, prototypes[i_prototype, :])

            # Find for which samples this prototype is the closes and has a different label
            ii_diff = i_prototype == i_dist_diff
            gradient[i_prototype] -= self._partial_gradient(ii_diff, relative_distance, dist_diff,
                                                            dist_same, data, prototypes[i_prototype, :])
        return gradient

    # S() = sum(f(mu(x)))
    def _cost(self, relative_distance):
        return np.sum(self.scaling(relative_distance))

    def __call__(self, variables, prototypes_labels, data, labels):
        # Variables 1D -> 2D prototypes
        prototypes = variables.reshape(self.prototypes_shape)

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
                                                             data, prototypes, prototypes_labels).ravel()


class RelevanceRelativeDistanceObjective(RelativeDistanceObjective):

    def __init__(self, distance=None, scaling=None, prototypes_shape=None, omega_shape=None):
        self.omega_shape = omega_shape
        super(RelevanceRelativeDistanceObjective, self).__init__(distance, scaling, prototypes_shape)

    def __call__(self, variables, prototype_labels, data, labels):
        num_features = data.shape[1]
        num_prototypes = prototype_labels.size

        prototypes_variables = variables[:(num_prototypes * num_features)]
        prototypes = prototypes_variables.reshape(prototype_labels.size, num_features)

        # TODO: Works for Global but not Local matrices...
        omega_variables = variables[(num_prototypes * num_features):]
        omega = omega_variables.reshape(self.omega_shape)

        # Update omega for the distance object used in compute_distance
        self.distance.omega = omega
        self.distance.normalise()
        dist_same, dist_diff, i_dist_same, i_dist_diff = self._compute_distance(data, labels,
                                                                                prototypes, prototype_labels)
        gradient = np.zeros(prototypes.shape)
        gradient_omega = np.zeros(omega.shape).ravel()

        for i_prototype in range(0, num_prototypes):
            ii_same = i_prototype == i_dist_same
            ii_diff = i_prototype == i_dist_diff

            relative_dist_same = np.atleast_2d(
                        self.scaling.gradient(_relative_distance(dist_same[ii_same], dist_diff[ii_same])) *
                        _relative_distance_grad(dist_same[ii_same], dist_diff[ii_same]))
            relative_dist_diff = np.atleast_2d(
                        self.scaling.gradient(_relative_distance(dist_same[ii_diff], dist_diff[ii_diff])) *
                        _relative_distance_grad(dist_diff[ii_diff], dist_same[ii_diff]))

            # Gradient prototypes TODO: Handle the case where there is no data... this prototype is never the closest with different label.
            grad_dist_same = np.atleast_2d(self.distance.gradient(data[ii_same, :], prototypes[i_prototype, :]))
            grad_dist_diff = np.atleast_2d(self.distance.gradient(data[ii_diff, :], prototypes[i_prototype, :]))

            if relative_dist_same.size > 0:
                gradient[i_prototype, :] = gradient[i_prototype, :] + relative_dist_same.dot(grad_dist_same)
            if relative_dist_diff.size > 0:
                gradient[i_prototype, :] = gradient[i_prototype, :] - relative_dist_diff.dot(grad_dist_diff)

            # Gradient Omega
            grad_omega_same = self.distance.omega_gradient(data[ii_same, :], prototypes[i_prototype, :])
            grad_omega_diff = self.distance.omega_gradient(data[ii_diff, :], prototypes[i_prototype, :])

            # Seems to give similar answers to Kerstins's matlab version on Iris dataset
            if grad_omega_same.size != 0:
                gradient_omega = gradient_omega + np.sum(np.atleast_2d(relative_dist_same).T * grad_omega_same, axis=0)
            if grad_omega_diff.size != 0:
                gradient_omega = gradient_omega - np.sum(np.atleast_2d(relative_dist_diff).T * grad_omega_diff, axis=0)

        return np.sum(self.scaling(_relative_distance(dist_same, dist_diff))), np.append(gradient.ravel(), gradient_omega)