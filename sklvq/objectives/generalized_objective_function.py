from .utils import _find_min
from . import ObjectiveBaseClass

import numpy as np


class GeneralizedObjectiveFunction(ObjectiveBaseClass):
    def __init__(self, activation=None, discriminant=None):
        self.activation = activation
        self.discriminant = discriminant

    def cost(self, data, labels, model):
        pass

    def gradient(self, data, labels, model):
        # Step 1: Compute distances between data and the model (how is depending on model and coupled distance function)
        distances = model.distance_(data, model)

        # Step 2: Find for all samples the distance between closest prototype with same label (d1) and different
        # label (d2). ii_same marks for all samples the prototype with the same label.
        ii_same = np.transpose(np.array([labels == prototype_label for prototype_label in model.prototypes_labels_]))

        # For each prototype mark the samples that have a different label
        ii_diff = ~ii_same

        # For each sample find the closest prototype with the same label. Returns distance and index of prototype
        dist_same, i_dist_same = _find_min(ii_same, distances)

        # For each sample find the closest prototype with a different label. Returns distance and index of prototype
        dist_diff, i_dist_diff = _find_min(ii_diff, distances)

        # Step 3: Evaluate for all samples the discriminant function: u(d1, d2) = (d1 - d2) / (d1 + d2)
        discriminant_score = self.discriminant(dist_same, dist_diff)

        # Allocation of the gradient
        # TODO: should be asked from model we don't know for which thing we compute the gradient here...
        gradient = np.zeros(model.variables_shape_)

        # For each prototype
        for i_prototype in range(0, model.prototypes_labels_.size):
            # Find for which samples it is the closest/winner AND has the same label
            ii_winner_same = (i_prototype == i_dist_same)
            if any(ii_winner_same):  # Only if these cases exist we can compute an update
                # magnitude is a scalar per sample
                magnitude = (self.activation.gradient(discriminant_score[ii_winner_same])) * \
                            (self.discriminant.gradient(dist_same[ii_winner_same],
                                                        dist_diff[ii_winner_same]))
                # direction is a vector per sample. NOTE Let the distance gradient function figure out what to do.
                direction = model.distance_.gradient(data[ii_winner_same], model, i_prototype)

                # The by magnitude weighted sum of directions, direction.
                gradient[i_prototype, :] -= (magnitude.dot(direction)).squeeze()

            # Find for which samples this prototype is the closest and has a different label
            ii_winner_diff = (i_prototype == i_dist_diff)
            if any(ii_winner_diff):
                # (dS / dmu * dmu / dd_K * dd_K / dw_K) K closest with different label
                magnitude = (self.activation.gradient(discriminant_score[ii_winner_diff])) * \
                            (self.discriminant.gradient(dist_diff[ii_winner_diff],
                                                        dist_same[ii_winner_diff]))
                # NOTE Let the distance gradient function figure out what to do.
                direction = model.distance_.gradient(data[ii_winner_diff], model, i_prototype)

                gradient[i_prototype, :] += (magnitude.dot(direction)).squeeze()

        return gradient
