from sklvq.objectives import ObjectiveBaseClass
from sklvq import activations, discriminants

import numpy as np

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass

ACTIVATION_FUNCTIONS = [
    "identity",
    "sigmoid",
    "soft-plus",
    "swish",
]

DISCRIMINANT_FUNCTIONS = [
    "relative-distance",
]


class GeneralizedLearningObjective(ObjectiveBaseClass):
    def __init__(
        self,
        activation_type: Union[type, str],
        activation_params: dict,
        discriminant_type: Union[type, str],
        discriminant_params: dict,
    ):
        self.activation = activations.grab(
            activation_type,
            class_kwargs=activation_params,
            whitelist=ACTIVATION_FUNCTIONS,
        )

        self.discriminant = discriminants.grab(
            discriminant_type,
            class_kwargs=discriminant_params,
            whitelist=DISCRIMINANT_FUNCTIONS,
        )

    def __call__(
        self,
        variables: np.ndarray,
        model: "LVQBaseClass",
        data: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        model._set_model_params(model._to_params(variables))

        dist_same, dist_diff, _, _ = _compute_distance(data, labels, model)

        return np.sum(self.activation(self.discriminant(dist_same, dist_diff)))

    def gradient(
        self,
        variables: np.ndarray,
        model: "LVQBaseClass",
        data: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        model._set_model_params(model._to_params(variables))

        dist_same, dist_diff, i_dist_same, i_dist_diff = _compute_distance(
            data, labels, model
        )

        discriminant_score = self.discriminant(dist_same, dist_diff)

        gradient = np.zeros(variables.size)

        # For each prototype
        for i_prototype in range(0, model.prototypes_labels_.size):
            # Find for which samples it is the closest/winner AND has the same label
            ii_winner_same = i_prototype == i_dist_same
            if any(ii_winner_same):
                # Only if these cases exist we can compute an update
                # Computes the following partial derivative: df/du
                activation_gradient = self.activation.gradient(
                    discriminant_score[ii_winner_same]
                )

                #  Computes the following partial derivatives: du/ddi, with i = 1
                discriminant_gradient = self.discriminant.gradient(
                    dist_same[ii_winner_same], dist_diff[ii_winner_same], True
                )

                # Computes the following partial derivatives: ddi/dwi, with i = 1
                distance_gradient = model._distance.gradient(
                    data[ii_winner_same], model, i_prototype
                )

                # The distance vectors weighted by the activation and discriminant partial
                # derivatives.
                gradient += (activation_gradient * discriminant_gradient).dot(
                    distance_gradient
                )

            # Find for which samples this prototype is the closest and has a different label
            ii_winner_diff = i_prototype == i_dist_diff
            if any(ii_winner_diff):
                # Computes the following partial derivative: df/du
                activation_gradient = self.activation.gradient(
                    discriminant_score[ii_winner_diff]
                )

                #  Computes the following partial derivatives: du/ddi, with i = 2
                discriminant_gradient = self.discriminant.gradient(
                    dist_same[ii_winner_diff], dist_diff[ii_winner_diff], False
                )

                # Computes the following partial derivatives: ddi/dwi, with i = 2
                distance_gradient = model._distance.gradient(
                    data[ii_winner_diff], model, i_prototype
                )

                # The distance vectors weighted by the activation and discriminant partial
                # derivatives.
                gradient += (activation_gradient * discriminant_gradient).dot(
                    distance_gradient
                )

        return gradient


def _find_min(indices: np.ndarray, distances: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Helper function to find the minimum distance and the index of this distance. """
    dist_temp = np.where(indices, distances, np.inf)
    # TODO: Optimize, finds min value/index twice.
    return dist_temp.min(axis=1), dist_temp.argmin(axis=1)


def _compute_distance(data: np.ndarray, labels: np.ndarray, model: "LVQBaseClass"):
    """ Computes the distances between each prototype and each observation and finds all indices
    where the shortest distance is that of the prototype with the same label and with a different label. """

    # Step 1: Compute distances between data and the model (how is depending on model and coupled
    # distance function)
    distances = model._distance(data, model)

    # Step 2: Find for all samples the distance between closest prototype with same label (d1)
    # and different label (d2). ii_same marks for all samples the prototype with the same label.
    ii_same = np.transpose(
        np.array(
            [labels == prototype_label for prototype_label in model.prototypes_labels_]
        )
    )

    # For each prototype mark the samples that have a different label
    ii_diff = ~ii_same

    # For each sample find the closest prototype with the same label. Returns distance and index
    # of prototype
    dist_same, i_dist_same = _find_min(ii_same, distances)

    # For each sample find the closest prototype with a different label. Returns distance and
    # index of prototype
    dist_diff, i_dist_diff = _find_min(ii_diff, distances)

    return dist_same, dist_diff, i_dist_same, i_dist_diff
