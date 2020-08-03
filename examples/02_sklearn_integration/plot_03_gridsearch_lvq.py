"""
===========
Grid Search
===========
"""

###############################################################################
# First, imports and load iris dataset

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklvq import GMLVQ

# Slightly more interesting dataset
data, labels = load_iris(return_X_y=True)

###############################################################################
# We first need to create a pipeline and initialize a parameter grid we want to search.

# Initialize the standardScaler (z-transform) object
standard_scaler = StandardScaler()

# Initialize the GMLVQ model
model = GMLVQ()

# Create pipeline that first scales the X and that inputs it to the GMLVQ model
pipeline = make_pipeline(standard_scaler, model)

# These are some of the relevant solver, distances and activation types for GMLVQ
solvers_types = [
    "steepest-gradient-descent",
    "waypoint-gradient-descent",
]

distance_types = ["adaptive-squared-euclidean"]

# We are using a pipeline so we need to prepend the parameters with the name of the
# class we want to provide the arguments to.
param_grid = [
    {
        "gmlvq__solver_type": solvers_types,
        "gmlvq__distance_type": distance_types,
        "gmlvq__activation_type": ["identity"],
    },
    {
        "gmlvq__solver_type": solvers_types,
        "gmlvq__distance_type": distance_types,
        "gmlvq__activation_type": ["sigmoid"],
        "gmlvq__activation_params": [{"beta": beta} for beta in range(2, 4, 1)],
    },
]

# Initialize a repeated stratiefiedKFold object
repeated_kfolds = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

# Provide everything to the GridsearchCV object from sklearn
search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="accuracy",
    cv=repeated_kfolds,
    n_jobs=4,
    return_train_score=False,
    verbose=10,
)

# Fit the X as one would with any other estimator.
search.fit(data, labels)

# Print the best CV score and parameters.
print("\nBest parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
