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


# Strategy
class AbstractObjective(ABC):

    def __init__(self, distance):
        self.distance = distance

    @abstractmethod
    def __call__(self, variables, prototype_labels, data, labels):
        raise NotImplementedError("You should implement this!")

    def compute_distance(self, data, labels, prototypes, prototype_labels):
        """ Computes the distances between each prototype and each observation and finds all indices where the shortest
        distance is that of the prototype with the same label and with a different label. """

        distances = self.distance(data, prototypes)

        ii_same = np.transpose(np.array([labels == prototype_label for prototype_label in prototype_labels]))
        ii_diff = ~ii_same

        dist_same, i_dist_same = _find_min(ii_same, distances)
        dist_diff, i_dist_diff = _find_min(ii_diff, distances)

        return dist_same, dist_diff, i_dist_same, i_dist_diff


class RelativeDistanceObjective(AbstractObjective):
    def __init__(self, distance=None, scaling=None, prototypes_shape=None):
        self.prototypes_shape = prototypes_shape
        self.scaling = scaling
        super(RelativeDistanceObjective, self).__init__(distance)

    def __call__(self, variables, prototype_labels, data, labels):
        # Initialisation
        num_prototypes = prototype_labels.size
        gradient = np.zeros(self.prototypes_shape)

        # Variables 1D -> 2D prototypes
        prototypes = variables.reshape(self.prototypes_shape)

        # dist_same, d_J(X, w_J), contains the distances between all
        # samples X and the closest prototype with the same label (_J)
        # i_dist_same tells you which prototype was closest.

        # dist_diff, d_K(X, w_K), contains the distances between all
        # samples X and the closest prototype with a different label (_K)
        # i_dist_diff same size as dist_diff, contains indices corresponding to the label of the closest prototype
        dist_same, dist_diff, i_dist_same, i_dist_diff = self.compute_distance(data, labels,
                                                                               prototypes, prototype_labels)

        for i_prototype in range(0, num_prototypes):
            # Find for which samples this prototype is the closest and has the same label
            ii_same = i_prototype == i_dist_same
            # Find for which samples this prototype is the closes and has a different label
            ii_diff = i_prototype == i_dist_diff

            # f'(u(x)) * u'_J(x) for all x (samples) in data (J: prototype label equals sample label)
            relative_dist_same = (  # All samples where the current prototype is the closest and has the same label
                        self.scaling.gradient(_relative_distance(dist_same[ii_same], dist_diff[ii_same])) *
                        _relative_distance_grad(dist_same[ii_same], dist_diff[ii_same]))
            # f'(u(x)) * u'_K(x) for all x (samples) in data (K: prototype label does not equal sample label)
            relative_dist_diff = (  # All samples where the current prototype is the closest and has a different label
                        self.scaling.gradient(_relative_distance(dist_diff[ii_diff], dist_same[ii_diff])) *
                        _relative_distance_grad(dist_diff[ii_diff], dist_same[ii_diff]))

            # d'_J(x, w) for all x (samples) in data and the current prototype
            grad_same = self.distance.gradient(data[ii_same, :], prototypes[i_prototype, :])
            # d'_K(x, w) for all x (samples) in data and the current prototype
            grad_diff = self.distance.gradient(data[ii_diff, :], prototypes[i_prototype, :])

            # Gradient for the current prototype: [f'(u(x)) * u'_J(x) * d'_J(x, w)] - [f'(u(x)) * u'_K(x) * d'_K(x, w)]
            gradient[i_prototype, :] = (relative_dist_same.dot(grad_same) - relative_dist_diff.dot(grad_diff))

        # First part is S = Sum(f(u(x))), Second part: gradient of all the prototypes "flattened" (ravel()) -> 1D array
        return np.sum(self.scaling(_relative_distance(dist_same, dist_diff))), gradient.ravel()


class RelevanceRelativeDistanceObjective(AbstractObjective):

    def __init__(self, distance=None, scaling=None, prototypes_shape=None, omega_shape=None):
        self.prototypes_shape = prototypes_shape
        self.omega_shape = omega_shape
        self.scaling = scaling
        super(RelevanceRelativeDistanceObjective, self).__init__(distance)

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
        dist_same, dist_diff, i_dist_same, i_dist_diff = self.compute_distance(data, labels,
                                                                               prototypes, prototype_labels)
        gradient = np.zeros(prototypes.shape)
        gradient_omega = np.zeros(omega.shape)

        for i_prototype in range(0, num_prototypes):
            ii_same = i_prototype == i_dist_same
            ii_diff = i_prototype == i_dist_diff

            relative_dist_same = (
                        self.scaling.gradient(_relative_distance(dist_same[ii_same], dist_diff[ii_same])) *
                        _relative_distance_grad(dist_same[ii_same], dist_diff[ii_same]))
            relative_dist_diff = (
                        self.scaling.gradient(_relative_distance(dist_diff[ii_diff], dist_same[ii_diff])) *
                        _relative_distance_grad(dist_diff[ii_diff], dist_same[ii_diff]))

            # Gradient prototypes
            grad_dist_same = self.distance.gradient(data[ii_same, :], prototypes[i_prototype, :])
            grad_dist_diff = self.distance.gradient(data[ii_diff, :], prototypes[i_prototype, :])

            gradient[i_prototype, :] = (relative_dist_same.dot(grad_dist_same) - relative_dist_diff.dot(grad_dist_diff))

            # Gradient Omega
            grad_omega_same = self.distance.omega_gradient(data[ii_same, :], prototypes[i_prototype, :])
            grad_omega_diff = self.distance.omega_gradient(data[ii_diff, :], prototypes[i_prototype, :])




        return np.sum(self.scaling(_relative_distance(dist_same, dist_diff))), gradient.ravel()