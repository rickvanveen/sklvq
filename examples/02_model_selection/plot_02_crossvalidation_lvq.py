"""
================
Generalizability
================

In all previous examples we looked at the training performance of the models. However, in practice it is much more
interesting how wel a model performs on unseen data (generalizability of the model).
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    cross_val_score,
    RepeatedKFold,
)

from sklvq import GMLVQ

data, labels = load_iris(return_X_y=True)

###############################################################################
# Cross validation
# ................
# Sklearn provides a very handy way of performing cross validation. For this purpose we firstly create a pipeline
# and initiate an sklearn object that will repeatedly create k folds for us.

scaler = StandardScaler()

model = GMLVQ(
    distance_type="adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="waypoint-gradient-descent",
    solver_params={"max_runs": 10, "k": 3, "step_size": np.array([0.1, 0.05])},
    random_state=1428,
)

pipeline = make_pipeline(scaler, model)

repeated_10_fold = RepeatedKFold(n_splits=10, n_repeats=10)

accuracy = cross_val_score(pipeline, data, labels, cv=repeated_10_fold)

print(
    "Accuracy, mean (std): {:.2f} ({:.2f})".format(
        np.mean(accuracy), np.std(accuracy)
    )
)
