from .utils import _find_min
from . import ObjectiveBaseClass

import numpy as np


class GeneralizedObjective(ObjectiveBaseClass):
    def __init__(self, activation=None, discriminant=None):
        self.activation = activation
        self.discriminant = discriminant

    # Computes all the distances such that we only need to do this once.
    @staticmethod
    def _compute_distance(data, labels, model):
        """ Computes the distances between each prototype and each observation and finds all indices where the shortest
        distance is that of the prototype with the same label and with a different label. """

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

        return dist_same, dist_diff, i_dist_same, i_dist_diff

    def cost(self, variables, model, data, labels):
        model.set_from_variables(variables)

        dist_same, dist_diff, _, _ = self._compute_distance(data, labels, model)

        return np.sum(self.activation(self.discriminant(dist_same, dist_diff)))


    # TODO: This function should be the models problem so we can still use this one class for multiple GXLVQ
    #  algorithms? However, do they take the same input.
    # d1, d2, u(d1, d2), x, (w_i, [omega]), i's)
    def _gradient(self, dist1, dist2, discriminant_score, data, model, i_prototype):
        # Using the notation from the original GLVQ paper (Sato and Yamada 1996)
        gradient = np.zeros(model.prototypes_.shape)

        # Computes the following partial derivative: df/du
        activation_gradient = self.activation.gradient(discriminant_score)

        #  Computes the following partial derivatives: du/ddi, with i = 1,2 (depending on order of input)
        discriminant_gradient = self.discriminant.gradient(dist1, dist2)

        # Computes the following partial derivatives: ddi/dwi, with i = 1,2(depending on input)
        distance_gradient = model.distance_.gradient(data, model, i_prototype)

        # Finally: dS/dwi, with i = 1,2
        gradient[i_prototype, :] = ((activation_gradient * discriminant_gradient).dot(distance_gradient)).squeeze()

        # Here it is in 2d shape now we have to put it to the correct 1d (only considering glvq)
        return gradient.ravel()

    # TODO: Compatibility with scipy lbfgs solver: (variables, model, data, labels)
    def gradient(self, variables, model, data, labels):
        model.set_from_variables(variables)

        dist_same, dist_diff, i_dist_same, i_dist_diff = self._compute_distance(data, labels, model)

        # Step 3: Evaluate for all samples the discriminant function: u(d1, d2) = (d1 - d2) / (d1 + d2)
        discriminant_score = self.discriminant(dist_same, dist_diff)  # TODO this should be u(x) not u(d1, d2)?

        # Allocation of the gradient
        gradient = np.zeros(model.variables_size_)

        # gradient = np.zeros(model.variables_size_)

        # For each prototype
        for i_prototype in range(0, model.prototypes_labels_.size):
            # Find for which samples it is the closest/winner AND has the same label
            ii_winner_same = (i_prototype == i_dist_same)
            if any(ii_winner_same):  # Only if these cases exist we can compute an update
                # Input is 1. distance to closest prototype with the same label (d1), 2. distance to closest
                # prototype with a different label (d2) 3. u(x, d1, d2), 4. relevant xi 5. model (w1, w2) 6. the
                # index of w1 and w2
                gradient += self._gradient(dist_same[ii_winner_same], dist_diff[ii_winner_same],
                                           discriminant_score[ii_winner_same], data[ii_winner_same, :],
                                           model, i_prototype)

            # Find for which samples this prototype is the closest and has a different label
            ii_winner_diff = (i_prototype == i_dist_diff)
            if any(ii_winner_diff):
                gradient -= self._gradient(dist_diff[ii_winner_diff], dist_same[ii_winner_diff],
                                           discriminant_score[ii_winner_diff], data[ii_winner_diff, :],
                                           model, i_prototype)

        return gradient
