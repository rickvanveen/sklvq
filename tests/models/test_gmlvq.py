import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import make_pipeline

from sklvq import GMLVQ


def test_gmlvq():
    iris = datasets.load_iris()

    estimator = GMLVQ()
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
    distance_types = ["adaptive-squared-euclidean"]

    # Every compatible activation
    activation_types = ["identity", "sigmoid", "soft-plus", "swish"]

    param_grid = [
        {
            "gmlvq__relevance_normalization": [True, False],
            "gmlvq__solver_type": solvers_types,
            "gmlvq__discriminant_type": discriminant_types,
            "gmlvq__distance_type": distance_types,
            "gmlvq__activation_type": activation_types,
        },
        {
            "gmlvq__relevance_normalization": [True, False],
            "gmlvq__solver_type": stochastic_solver_types,
            "gmlvq__solver_params": [
                {"batch_size": 0, "step_size": np.array([0.1, 0.001])},
                {"batch_size": 1, "step_size": [0.1, 0.01]},
                {"batch_size": 2, "step_size": (0.1, 0.01)},
            ],
            "gmlvq__discriminant_type": discriminant_types,
            "gmlvq__distance_type": distance_types,
            "gmlvq__activation_type": activation_types,
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

    # print(f"\nBest parameter (CV score={search.best_score_:0.3f}):")
    # print(search.best_params_)


def test_gmlvq_():
    X, y = datasets.load_iris(return_X_y=True)
    model = GMLVQ().fit(X, y)

    assert np.all(np.isclose(model.lambda_, GMLVQ._compute_lambda(model.omega_hat_)))
    assert np.all(np.isclose(np.linalg.norm(model.eigenvectors_, axis=1), 1.0))

def test_relevance_correction():
    
    X, y = datasets.load_iris(return_X_y=True)
    model = GMLVQ(random_state=1048).fit(X, y)

    leading_eigenvector = model.eigenvectors_[0]

    correction_vectors = np.atleast_2d(leading_eigenvector)

    correction_matrix = np.identity(correction_vectors.shape[1]) - (
        correction_vectors.T.dot(correction_vectors)
    )
    # Check if it is a symmetric matrix
    assert np.allclose(correction_matrix, correction_matrix.T)

    model2 = GMLVQ(relevance_correction=correction_matrix, random_state=1048).fit(X, y)

    # Check that omega from the corrected model does not contain contribution from the vectors we wanted to remove
    assert np.allclose(np.dot(model2.omega_, correction_vectors.T.dot(correction_vectors)), np.zeros(correction_vectors.shape[1]))



       



