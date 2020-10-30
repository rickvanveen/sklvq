"""
================
Cross validation
================

In all previous examples we showed the training performance of the models. However,
in practice it is much more interesting how well a model performs on
unseen data, i.e., the generalizability of the model. We can use `crossvalidation`_ for this.

.. _crossvalidation: https://scikit-learn.org/stable/modules/cross_validation.html
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    cross_val_score,
    RepeatedKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklvq import GMLVQ

data, labels = load_iris(return_X_y=True)

###############################################################################
# Sklearn provides a very handy way of performing cross validation. For this
# purpose we firstly create a pipeline and initiate a sklearn object that will
# repeatedly create k folds for us.

# Create a scaler instance
scaler = StandardScaler()

# Create a GMLVQ  model instance
model = GMLVQ(
    distance_type="adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="waypoint-gradient-descent",
    solver_params={"max_runs": 10, "k": 3, "step_size": np.array([0.1, 0.05])},
    random_state=1428,
)

# Link them together (Note this will work as it should in a CV setting, i.e., it's fitted to the
# training data and predict is used for the testing data which makes sure the test data is
# scaled using the tranformation parameters found during training.
pipeline = make_pipeline(scaler, model)

# Create an object that n_repeat times creates k=10  folds.
repeated_10_fold = RepeatedKFold(n_splits=10, n_repeats=10)

# Call the cross_val_score using all created instances and loaded data. Note it can accept
# different and also multiple scoring parameters
accuracy = cross_val_score(pipeline, data, labels, cv=repeated_10_fold, scoring="accuracy")

# Print the mean and standard deviation of the cross validation testing scores.
print(
    "Accuracy, mean (std): {:.2f} ({:.2f})".format(np.mean(accuracy), np.std(accuracy))
)
