"""
======================
Discriminant Functions
======================
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

from sklvq import GLVQ
from sklvq.discriminants import DiscriminantBaseClass

data, labels = load_iris(return_X_y=True)

###############################################################################
# The sklvq package contains a single discriminant function and additions are very welcome. Note
# that they should work with the sklvq.objectives.GeneralizedLearningObjective, i.e.,
# passing additional or different arguments is not possible.


# The discriminative function is depended on the objective function. This determines the
# parameters of the call and gradient. See sklvq.objective.GeneralizedLearningObjective.
class CustomRelativeDistance(DiscriminantBaseClass):
    def __call__(self, dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
        # dist_same = distance to prototype with same label as X.
        # dist_diff = distance to prototype with different label as X.
        return (dist_same - dist_diff) / (dist_same + dist_diff)

    def gradient(
        self, dist_same: np.ndarray, dist_diff: np.ndarray, winner_same: bool
    ) -> np.ndarray:
        # Winner_same is an boolean flag to indicate if the considered prototype has the same or
        # a different label compared to the considered X.
        if winner_same:
            return _gradient_same(dist_same, dist_diff)
        return _gradient_diff(dist_same, dist_diff)


# Gradient depends on if the label is the same or different
def _gradient_same(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    return 2 * dist_diff / (dist_same + dist_diff) ** 2


# Gradient depends on if the label is the same or different
def _gradient_diff(dist_same: np.ndarray, dist_diff: np.ndarray) -> np.ndarray:
    return -2 * dist_same / (dist_same + dist_diff) ** 2


###############################################################################
# The CustomRelativeDistance above, accompanied with some tests and documentation, would make a
# great addition to the sklvq package. However, it can also directly be passed to the algorithm.

model = GLVQ(
    discriminant_type=CustomRelativeDistance,
    distance_type="squared-euclidean",
    activation_type="sigmoid",
    activation_params={"beta": 2},
)

model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))
