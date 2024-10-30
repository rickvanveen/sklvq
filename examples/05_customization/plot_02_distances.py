"""
==================
Distance Functions
==================
"""

from typing import TYPE_CHECKING

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import pairwise_distances

from sklvq.distances import DistanceBaseClass

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass

from sklvq import GLVQ

data, labels = load_iris(return_X_y=True)

###############################################################################
# The sklvq contains already a few distance function. Please see the API reference under
# Documentation. It has a very similar base class to that of the activation functions.
# However, the structure in which the distance and especially the gradient with respect
# to the different parameters need to be returned are important. Furthermore not every
# distance functions works with every algorithm. Below the
# `sklvq.distances.SquaredEuclidean`, which is suitable for the GLVQ algorithm.


class CustomSquaredEuclidean(DistanceBaseClass):
    # The distance implementations use the sklearn pairwise distance function.
    def __init__(self, **other_kwargs):
        self.metric_kwargs = {"metric": "euclidean", "squared": True}

        if other_kwargs is not None:
            self.metric_kwargs.update(other_kwargs)

    # The call function needs to return a matrix with the number of X points on the
    # rows and the columns the distance to the prototypes.
    def __call__(self, data: np.ndarray, model: "LVQBaseClass") -> np.ndarray:
        return pairwise_distances(
            data,
            model.prototypes_,
            **self.metric_kwargs,
        )

    # The gradient is slightly more difficult as the gradient (with respect to 1
    # prototype) needs to be provided in a vector the size of all the prototypes.
    # Hence, all values are zero except those of the prototype indicated by the index
    # i_prototype. In the case of GMLVQ and LGMVLQ distance functions als the gradient
    # of the omega matrix needs to be returned (in this same vector). See the API
    # reference under Documentation or github for other distance functions and their
    # implementation.
    def gradient(self, data: np.ndarray, model: "LVQBaseClass", i_prototype: int) -> np.ndarray:
        prototypes = model.get_model_params()
        (num_samples, num_features) = data.shape

        distance_gradient = np.zeros((num_samples, prototypes.size))

        ip_start = i_prototype * num_features
        ip_end = ip_start + num_features

        distance_gradient[:, ip_start:ip_end] = -2 * (data - prototypes[i_prototype, :])

        return distance_gradient


###############################################################################
# The CustomSquaredEuclidean above, accompanied with some tests and documentation, would make a
# great addition to the sklvq package. However, it can also directly be passed to
# the algorithm.

model = GLVQ(
    distance_type=CustomSquaredEuclidean,
    activation_type="sigmoid",
    activation_params={"beta": 2},
)

model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))
