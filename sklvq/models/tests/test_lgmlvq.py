import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import make_pipeline

from .. import LGMLVQ


def test_lgmlvq():
    iris = datasets.load_iris()

    estimator = LGMLVQ()
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
    distance_types = ["local-adaptive-squared-euclidean"]

    # Every compatible activation
    activation_types = ["identity", "sigmoid", "soft-plus", "swish"]

    param_grid = [
        {
            "lgmlvq__solver_type": solvers_types,
            "lgmlvq__discriminant_type": discriminant_types,
            "lgmlvq__distance_type": distance_types,
            "lgmlvq__activation_type": activation_types,
        }
    ]

    repeated_kfolds = RepeatedStratifiedKFold(n_splits=2, n_repeats=1)

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="accuracy",
        cv=repeated_kfolds,
        return_train_score=True,
    )

    search.fit(iris.data, iris.target)

    assert np.all(search.cv_results_["mean_train_score"] > 0.75)
    assert np.all(search.cv_results_["mean_test_score"] > 0.75)

    print("\nBest parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
