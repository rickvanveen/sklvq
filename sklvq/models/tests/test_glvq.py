import numpy as np
import pytest
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import make_pipeline

from sklvq.activations._identity import Identity

from .. import GLVQ


def test_glvq_hyper_parameters():
    X, y = datasets.load_iris(return_X_y=True)

    # Prototype initialization
    with pytest.raises(ValueError):
        GLVQ(prototype_params={"prototypes_per_class": np.array([1, 1])}).fit(X, y)

    with pytest.raises(ValueError):
        GLVQ(prototype_params={"prototypes_per_class": np.array([1, 0, 1])}).fit(X, y)

    with pytest.raises(ValueError):
        GLVQ(prototype_init="abc").fit(X, y)

    m = GLVQ(prototype_params={"prototypes_per_class": np.array([1, 2, 1])}).fit(X, y)
    assert m.prototypes_.shape[0] == 4

    # Activation string which does not exist
    with pytest.raises(ValueError):
        GLVQ(activation_type="abc123").fit(X, y)

    # Activation object instead of type
    activation_type = Identity()
    with pytest.raises(ValueError):
        GLVQ(activation_type=activation_type).fit(X, y)

    activation_type = Identity
    with pytest.raises(TypeError):
        GLVQ(activation_type=activation_type, activation_params={"beta": 0}).fit(X, y)

    m = GLVQ(activation_type=activation_type).fit(X, y)

    p = m.prototypes_
    m.set_model_params(np.random.random(size=(3, 4)))
    assert np.shares_memory(p, m.get_variables())
    assert np.all(m.get_variables() == m.prototypes_.ravel())

    model_params = m.to_model_params_view(m.get_variables())
    assert np.all(m.prototypes_.shape == model_params.shape)
    assert np.shares_memory(m.prototypes_, m.get_variables())
    assert np.shares_memory(model_params, m.get_variables())


def test_glvq():
    iris = datasets.load_iris()

    estimator = GLVQ(random_state=31415)
    pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)

    scipy_solvers_types = [
        "lbfgs",
        "bfgs"
    ]
    # Run each solver ones
    solvers_types = [
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
            "glvq__solver_type": scipy_solvers_types,
            "glvq__solver_params": [{"jac": None}, {}],
            "glvq__discriminant_type": discriminant_types,
            "glvq__distance_type": distance_types,
            "glvq__activation_type": activation_types,
        },
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
