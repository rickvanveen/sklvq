from . import DiscriminativeBaseClass

import numpy as np


# TODO: accept data rather than something preprocessed... discriminant(xi) should provide some score.
class RelativeDistance(DiscriminativeBaseClass):

    # TODO: Look into why runtime warming occurs and what to do about it. Maybe the initialization of the
    #  prototypes is too close together?
    def __call__(self, dist_same, dist_diff):
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
            return (dist_same - dist_diff) / (dist_same + dist_diff)

    def gradient(self, dist_same, dist_diff):
        """ The partial derivative of the objective itself with respect to the prototypes (GLVQ) """
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
            return 2 * dist_diff / (dist_same + dist_diff) ** 2

    def call(self, data, model):
        pass


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