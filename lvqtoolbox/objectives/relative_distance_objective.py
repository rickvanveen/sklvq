from .utils import _find_min
from . import ObjectiveBaseClass

import numpy as np


class RelativeDistanceObjective(ObjectiveBaseClass):
    def __init__(self, distance=None, activation=None):
        self.distance = distance
        self.activation = activation

    @staticmethod
    def _evaluate(dist_same, dist_diff):
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
            return (dist_same - dist_diff) / (dist_same + dist_diff)

    def evaluate(self, data, labels, prototypes, prototype_labels):
        pass

    def _prototype_gradient(self):
        pass

    @staticmethod
    def _gradient(dist_same, dist_diff):
        """ The partial derivative of the objective itself with respect to the prototypes (GLVQ) """
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
            return 2 * dist_diff / (dist_same + dist_diff) ** 2

    def gradient(self, data, labels, prototypes, prototype_labels):
        distances = self.distance(data, prototypes)

        # For each prototype mark the samples that have the same label
        ii_same = np.transpose(np.array([labels == prototype_label for prototype_label in prototype_labels]))

        # For each prototype mark the samples that have a different label
        ii_diff = ~ii_same

        # For each sample find the closest prototype with the same label. Returns distance and index of prototype
        dist_same, i_dist_same = _find_min(ii_same, distances)

        # For each sample find the closest prototype with a different label. Returns distance and index of prototype
        dist_diff, i_dist_diff = _find_min(ii_diff, distances)

        # Evaluate for all samples xi in X the objective function: u(xi) = (d1 - d2) / (d1 + d2)
        relative_distance = self._evaluate(dist_same, dist_diff)

        # Allocation of the gradient
        gradient = np.zeros(prototypes.shape)

        # For each prototype
        for i_prototype in range(0, prototype_labels.size):
            # Find for which samples it is the closest/winner AND has the same label
            ii_winner_same = (i_prototype == i_dist_same)
            if any(ii_winner_same):  # Only if these cases exist we can compute an update
                # magnitude is a scalar per sample
                magnitude = (self.activation.gradient(relative_distance[ii_winner_same])) * \
                            (self._gradient(dist_same[ii_winner_same], dist_diff[ii_winner_same]))
                # direction is a vector per sample
                direction = self.distance.gradient(data[ii_winner_same], prototypes[i_prototype, :])

                # The by magnitude weighted sum of directions, direction.
                gradient[i_prototype, :] -= (magnitude.dot(direction)).squeeze()

            # Find for which samples this prototype is the closest and has a different label
            ii_winner_diff = (i_prototype == i_dist_diff)
            if any(ii_winner_diff):
                # (dS / dmu * dmu / dd_K * dd_K / dw_K) K closest with different label
                magnitude = (self.activation.gradient(relative_distance[ii_winner_diff])) * \
                            (self._gradient(dist_diff[ii_winner_diff], dist_same[ii_winner_diff]))
                direction = self.distance.gradient(data[ii_winner_diff], prototypes[i_prototype, :])

                gradient[i_prototype, :] += (magnitude.dot(direction)).squeeze()

        return gradient
