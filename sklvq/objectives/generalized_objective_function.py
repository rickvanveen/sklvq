from .utils import _find_min
from . import ObjectiveBaseClass

import numpy as np


class GeneralizedObjectiveFunction(ObjectiveBaseClass):
    def __init__(self, activation=None, discriminant=None):
        self.activation = activation
        self.discriminant = discriminant

    # TODO: separate this out as the discriminative function
    @staticmethod
    def _evaluate(dist_same, dist_diff):
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
            return (dist_same - dist_diff) / (dist_same + dist_diff)

    def evaluate(self, data, labels, model):
        pass

    # TODO: separate this out as the discriminative function's gradient
    @staticmethod
    def _gradient(dist_same, dist_diff):
        """ The partial derivative of the objective itself with respect to the prototypes (GLVQ) """
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
            return 2 * dist_diff / (dist_same + dist_diff) ** 2

    def gradient(self, data, labels, model):
        # TODO: Should be ask to the model... so this still works for GMLVQ. "model.distance(data)"
        distances = model.distance_(data, model.prototypes_)

        # For each prototype mark the samples that have the same label
        ii_same = np.transpose(np.array([labels == prototype_label for prototype_label in model.prototypes_labels_]))

        # For each prototype mark the samples that have a different label
        ii_diff = ~ii_same

        # For each sample find the closest prototype with the same label. Returns distance and index of prototype
        dist_same, i_dist_same = _find_min(ii_same, distances)

        # For each sample find the closest prototype with a different label. Returns distance and index of prototype
        dist_diff, i_dist_diff = _find_min(ii_diff, distances)

        # Evaluate for all samples xi in X the objective function: u(xi) = (d1 - d2) / (d1 + d2)
        relative_distance = self._evaluate(dist_same, dist_diff)
        # TODO: depends on discriminative function used "model.discriminative(xi)" Model has all the information as
        #  long it returns the right thing. Some score...

        # Allocation of the gradient
        # TODO: should be asked from model we don't know for which thing we compute the gradient here...
        gradient = np.zeros(model.prototypes_.shape)

        # For each prototype
        for i_prototype in range(0, model.prototypes_labels_.size):
            # Find for which samples it is the closest/winner AND has the same label
            ii_winner_same = (i_prototype == i_dist_same)
            if any(ii_winner_same):  # Only if these cases exist we can compute an update
                # magnitude is a scalar per sample
                magnitude = (self.activation.gradient(relative_distance[ii_winner_same])) * \
                            (self.discriminant.gradient(dist_same[ii_winner_same],
                                            dist_diff[ii_winner_same]))  # TODO: Depends on discriminative function used
                # direction is a vector per sample
                direction = model.distance_.gradient(data[ii_winner_same], model.prototypes_[i_prototype, :])

                # The by magnitude weighted sum of directions, direction.
                gradient[i_prototype, :] -= (magnitude.dot(direction)).squeeze()

            # Find for which samples this prototype is the closest and has a different label
            ii_winner_diff = (i_prototype == i_dist_diff)
            if any(ii_winner_diff):
                # (dS / dmu * dmu / dd_K * dd_K / dw_K) K closest with different label
                magnitude = (self.activation.gradient(relative_distance[ii_winner_diff])) * \
                            (self.discriminant.gradient(dist_diff[ii_winner_diff],
                                            dist_same[ii_winner_diff]))  # Todo: Depends on discriminative function used
                direction = model.distance_.gradient(data[ii_winner_diff], model.prototypes_[i_prototype, :])

                gradient[i_prototype, :] += (magnitude.dot(direction)).squeeze()

        return gradient
