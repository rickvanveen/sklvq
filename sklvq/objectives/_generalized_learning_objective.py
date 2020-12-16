import numpy as np

from ..objectives._base import ObjectiveBaseClass
from .. import activations, discriminants

from typing import Union
from typing import TYPE_CHECKING

from .._utils import init_class

if TYPE_CHECKING:
    from ..models import LVQBaseClass


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
    """Generalized learning objective

    Class that holds the generalized learning objective function and its gradient as described
    in `[1]`_.

    Parameters
    ----------
    activation_type : {"identity", "sigmoid", "soft-plus", "swish"} or type
        If string needs to be one of the indicated options. If not a string needs to be a custom
        activation class. See :class:`sklvq.activations.ActivationBaseClass`.
    activation_params : dict or None
        The dictionary with the parameters for the activation function or None if it doesn't
        require any parameters.
    discriminant_type: {"relative-distance"} or type
        Can only be the relative distance. If not a string it can be a custom class.
        See :class:`sklvq.discriminants.DiscriminantBaseClass`.
    discriminant_params : dict or None
        The dictionary with the parameters for the discriminant function or None if it doesn't
        require any parameters.

    Notes
    -----
    Compatible and used within the following models: :class:`.GLVQ`, :class:`.GMLVQ`,
    and :class:`.LGMLVQ`.

    References
    ----------
    _`[1]` Sato, A., and Yamada, K. (1996) "Generalized Learning Vector Quantization."
    Advances in Neural Network Information Processing Systems, 423–429, 1996."""

    def __init__(
        self,
        activation_type: Union[str, type],
        activation_params: dict,
        discriminant_type: Union[str, type],
        discriminant_params: dict,
    ):
        if activation_params is None:
            activation_params = {}

        activation_class = init_class(
            activations, activation_type, ACTIVATION_FUNCTIONS
        )
        self.activation = activation_class(**activation_params)

        if discriminant_params is None:
            discriminant_params = {}

        discriminant_class = init_class(
            discriminants, discriminant_type, DISCRIMINANT_FUNCTIONS
        )
        self.discriminant = discriminant_class(**discriminant_params)

    def __call__(
        self,
        model: "LVQBaseClass",
        data: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        r"""Computes the generalized learning objective:

            .. math::

                E = \sum_{i=1}^{N} f(\mu(d_0(\mathbf{x}_i), d_1(\mathbf{x}_i))

        with :math:`\mu(\cdot)` the discriminative function, :math:`f(\cdot)` the activation
        function, and :math:`d_0(\mathbf{x}_i)` and :math:`d_1(\mathbf{x}_i)` the shortest
        distance to a prototype  with  a different and the same label respectively.

        Parameters
        ----------
        model : LVQBaseClass
            The model which can be any LVQBaseClass compatible with this objective function.

        data: ndarray with shape (n_samples, n_features)
            The data.

        labels: ndarray with shape (n_samples)
            The labels of the samples in the data.

        Returns
        -------
        float:
            The cost
        """
        dist_same, dist_diff, _, _ = _compute_distance(data, labels, model)

        return np.sum(self.activation(self.discriminant(dist_same, dist_diff)))

    def gradient(
        self,
        model: "LVQBaseClass",
        data: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        r"""Computes the generalized learning objective's gradient with respect to the
        prototype with a different label:

            .. math::
                \frac{\partial E}{\partial \mathbf{w}_0} = \frac{\partial f}{\partial \mu}
                \frac{\partial \mu}{\partial d_0} \frac{\partial d_0}{\partial \mathbf{w}_0}

        with :math:`\mathbf{w}_0` the prototype with a different label than the data and :math:`d_0`
        the distance to that prototype.

            .. math::
                 \frac{\partial E}{\partial \mathbf{w}_1} = \frac{\partial f}{\partial \mu}
                 \frac{\partial \mu}{\partial d_1} \frac{\partial d_1}{\partial \mathbf{w}_1}

        with :math:`\mathbf{w}_1` the prototype with the same label as the data and :math:`d_1`
        the distance to that prototype.

        Parameters
        ----------
        model : LVQBaseClass
            The model which can be any LVQBaseClass compatible with this objective function.

        data: ndarray with shape (n_samples, n_features)
            The data.

        labels: ndarray with shape (n_samples)
            The labels of the samples in the data.

        Returns
        -------
        ndarray with the same shape as the model variables array (depending on the model)
            The generalized learning objective function's gradient

        """

        dist_same, dist_diff, i_dist_same, i_dist_diff = _compute_distance(
            data, labels, model
        )
        discriminant_score = self.discriminant(dist_same, dist_diff)

        # Pre-allocation, needs to be zero.
        gradient_buffer = np.zeros(model.get_variables().size)

        # For each prototype
        for i_prototype in range(0, model.prototypes_labels_.size):
            # Find for which samples it is the closest/winner AND has the same label
            # ii_winner_same = i_prototype == i_dist_same
            if i_prototype in i_dist_same:
                ii_winner_same = i_prototype == i_dist_same
                # Only if these cases exist we can/should compute an update
                self._partial_gradient(
                    gradient_buffer,
                    discriminant_score[ii_winner_same],
                    dist_same[ii_winner_same],
                    dist_diff[ii_winner_same],
                    True,
                    data[ii_winner_same, :],
                    model,
                    i_prototype,
                )

            # Find for which samples this prototype is the closest and has a different label
            if i_prototype in i_dist_diff:
                ii_winner_diff = i_prototype == i_dist_diff
                self._partial_gradient(
                    gradient_buffer,
                    discriminant_score[ii_winner_diff],
                    dist_same[ii_winner_diff],
                    dist_diff[ii_winner_diff],
                    False,
                    data[ii_winner_diff, :],
                    model,
                    i_prototype,
                )

        return gradient_buffer

    def _partial_gradient(
        self,
        gradient_buffer,
        discriminant_score,
        dist_same,
        dist_diff,
        winner_same,
        data,
        model,
        i_prototype,
    ):
        # Computes the following partial derivative: df/du
        activation_gradient = self.activation.gradient(discriminant_score)

        #  Computes the following partial derivatives: du/ddi, with i = 2
        discriminant_gradient = self.discriminant.gradient(
            dist_same, dist_diff, winner_same
        )

        # Computes the following partial derivatives: ddi/dwi, with i = 2
        distance_gradient = model._distance.gradient(data, model, i_prototype)

        # The distance vectors weighted by the activation and discriminant partial
        # derivatives.
        model.add_partial_gradient(
            gradient_buffer,
            (activation_gradient * discriminant_gradient).dot(distance_gradient),
            i_prototype,
        )


def _find_min(indices: np.ndarray, distances: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Helper function to find the minimum distance and the index of this distance. """
    # Set the irrelevant distances to infinity.
    dist_temp = np.where(indices, distances, np.inf)
    # Find the indices of the closest prototype (column)
    i_dist_min = dist_temp.argmin(axis=1)
    # Return the shortest distances and the indices of the prototypes.
    return dist_temp[np.arange(i_dist_min.size), i_dist_min], i_dist_min


def _compute_distance(data: np.ndarray, labels: np.ndarray, model: "LVQBaseClass"):
    """Computes the distances between each prototype and each observation and finds all indices
    where the shortest distance is that of the prototype with the same label and with a different label."""
    prototypes_labels = model.prototypes_labels_

    # Step 1: Compute distances between X and the model (how is depending on model and coupled
    # distance function)
    distances = model._distance(data, model)

    # Step 2: Find for all samples the distance between closest prototype with same label (d1)
    # and different label (d2). ii_same marks for all samples the prototype with the same label.

    num_samples = labels.size
    num_prototypes = model.prototypes_labels_.size
    if num_samples == 1:
        # Faster if num_samples == 1
        ii_same = np.atleast_2d(labels == prototypes_labels)
    elif num_samples < num_prototypes:
        # Faster to go over the labels if there are less than the prototypes.
        ii_same = np.array([label == prototypes_labels for label in labels])
    else:
        # List comprehension of the prototypes. This are all slight improvements to computation
        # time, as list comprehension takes quite some time.
        ii_same = np.transpose(
            [labels == prototype_label for prototype_label in prototypes_labels]
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
