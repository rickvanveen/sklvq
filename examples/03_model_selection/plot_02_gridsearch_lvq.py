"""
===========
Grid Search
===========

Cross validation  is not the whole story as it only can tell you  the expected performance of a
single set of (hyper) parameters. Luckily  sklearn also provides a way of trying out multiple
settings and return the CV scores for each of them. We can use `gridsearch`_ for this.

.. _gridsearch: https://scikit-learn.org/stable/modules/grid_search.html
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklvq import GMLVQ
data, labels = load_iris(return_X_y=True)

###############################################################################
# We first need to create a pipeline and initialize a parameter grid we want to search.

# Create the standard scaler instance
standard_scaler = StandardScaler()

# Create the GMLVQ model instance
model = GMLVQ()

# Link them together by using sklearn's pipeline
pipeline = make_pipeline(standard_scaler, model)

# We want to see the difference in performance of the two following solvers
solvers_types = [
    "steepest-gradient-descent",
    "waypoint-gradient-descent",
]

# Currently, the  sklvq package contains only the following distance function compatible with
# GMLVQ. However, see the customization examples for writing your own.
distance_types = ["adaptive-squared-euclidean"]

# Because we are using a pipeline we need to prepend the parameters with the name of the
# class of instance we want to provide the parameters for.
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
        "gmlvq__activation_params": [{"beta": beta} for beta in range(1, 4, 1)],
    },
]
# This grid can be read as: for each solver, try each distance type with the identity function,
# and the sigmoid activation function for each beta in the range(1, 4, 1)

# Initialize a repeated stratiefiedKFold object
repeated_kfolds = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

# Initilialize the gridsearch CV instance that will fit the pipeline (standard scaler, GMLVQ) to
# the data for each of the parameter sets in the grid. Where each fit is a 5 times
# repeated stratified 5 fold cross validation. For each set return the testing accuracy.
search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="accuracy",
    cv=repeated_kfolds,
    n_jobs=4,
    return_train_score=False,
    verbose=10,
)

# The gridsearch object can be fitted to the data.
search.fit(data, labels)

# Print the best CV score and parameters.
print("\nBest parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#
