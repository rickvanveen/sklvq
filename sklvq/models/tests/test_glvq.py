import pytest
import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import make_pipeline

from .. import GLVQ

# TODO: Test INIT


def test_glvq():
    iris = datasets.load_iris()

    estimator = GLVQ(random_state=31415)
    pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)

    # Run each solver ones
    solvers_types = [
        "lbfgs",
        "bfgs",
        "steepest-gradient-descent",
        "waypoint-gradient-descent",
        "adaptive-moment-estimation",
    ]
    discriminant_types = ["relative-distance"]

    # Every compatible distance
    distance_types = ["squared-euclidean", "euclidean"]

    # Every compatible activation
    activation_types = ["identity", "sigmoid", "soft-plus", "swish"]

    param_grid = [
        {
            "glvq__solver_type": solvers_types,
            "glvq__discriminant_type": discriminant_types,
            "glvq__distance_type": distance_types,
            "glvq__activation_type": activation_types,
        }
    ]

    repeated_kfolds = RepeatedStratifiedKFold(n_splits=2, n_repeats=1)

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=["accuracy", "roc_auc_ovo", "precision_macro", "recall_macro"],
        cv=repeated_kfolds,
        return_train_score=True,
        refit="roc_auc_ovo",
    )

    search.fit(iris.data, iris.target)

    assert np.all(search.cv_results_["mean_train_roc_auc_ovo"] > 0.75)
    assert np.all(search.cv_results_["mean_test_roc_auc_ovo"] > 0.75)

    print("\nBest parameter (CV roc_auc=%0.3f):" % search.best_score_)
    print(search.best_params_)