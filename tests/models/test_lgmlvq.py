import numpy as np
import pytest
from sklearn import datasets, preprocessing
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import make_pipeline

from sklvq import LGMLVQ
from tests.distances._utils import finite_difference


@pytest.mark.parametrize("localization", ["class", "prototypes"])
def test_objective_gradient_matrix_selection(localization):
    """Check distance derivatives and their accumulation into the real model's flat parameter buffer."""
    prototypes = np.array([[0, 0], [3, 0], [0, 4], [3, 4]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    factors = np.array([[[1, 0], [0, 2]], [[2, 0], [0, 1]], [[3, 0], [0, 1]], [[1, 0], [0, 1]]])

    # Initialize real model state, stopping before training can change the controlled example.
    model = LGMLVQ(
        prototype_n_per_class=2,
        relevance_localization=localization,
        relevance_normalization=False,
        activation_type="identity",
        random_state=0,
        solver_params={"callback": lambda state: True},
    ).fit(prototypes, labels)
    model.set_model_params((prototypes, factors[:2] if localization == "class" else factors))

    # Multiple winners exercise both shared class factors and distinct same-class prototype factors.
    data = np.array([[1, 2], [-2, 1], [4, 3]], dtype=float)
    targets = np.array([0, 0, 1])
    expected_gradient = finite_difference(lambda: model._objective(model, data, targets), [model.get_variables()])[0]
    actual_gradient = model._objective.gradient(model, data, targets)

    assert actual_gradient.shape == expected_gradient.shape
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-7)


def test_lgmlvq_hyper_params():
    X, y = datasets.load_iris(return_X_y=True)

    model = LGMLVQ(prototype_n_per_class=6, relevance_localization="prototypes").fit(X, y)
    assert model.omega_.shape[0] == (model.classes_.size * 6)

    model = LGMLVQ(prototype_n_per_class=6, relevance_localization="class").fit(X, y)
    assert model.omega_.shape[0] == model.classes_.size

    with pytest.raises(ValueError):
        LGMLVQ(prototype_n_per_class=6, relevance_localization="abc").fit(X, y)

    with pytest.raises(ValueError):
        LGMLVQ(prototype_n_per_class=6, relevance_localization=8).fit(X, y)


def test_lgmlvq():
    iris = datasets.load_iris()

    estimator = LGMLVQ()
    pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)

    # Run each solver ones
    solvers_types = [
        "lbfgs",
        "bfgs",
        "waypoint-gradient-descent",
        "adaptive-moment-estimation",
    ]
    stochastic_solver_types = [
        "steepest-gradient-descent",
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
        },
        {
            "lgmlvq__relevance_normalization": [True, False],
            "lgmlvq__solver_type": stochastic_solver_types,
            "lgmlvq__solver_params": [
                {"batch_size": 0, "step_size": np.array([0.1, 0.001])},
                {"batch_size": 1, "step_size": [0.1, 0.01]},
                {"batch_size": 2, "step_size": (0.1, 0.01)},
            ],
            "lgmlvq__discriminant_type": discriminant_types,
            "lgmlvq__distance_type": distance_types,
            "lgmlvq__activation_type": activation_types,
        },
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

    # print("\nBest parameter (CV score=%0.3f):" % search.best_score_)
    # print(search.best_params_)


def test_lgmlvq_relevances():
    X, y = datasets.load_iris(return_X_y=True)
    model = LGMLVQ().fit(X, y)

    assert np.all(np.isclose(model.lambda_, LGMLVQ._compute_lambdas(model.omega_hat_)))
    assert np.all(np.isclose(np.linalg.norm(model.eigenvectors_, axis=1), 1.0))
