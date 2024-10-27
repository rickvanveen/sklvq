import pytest
import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

from sklvq.models import GLVQ


class ProgressLogger:
    def __init__(self, k=4):
        self.states = np.array([])
        self.counter = 0
        self.k = k

    def __call__(self, state) -> bool:
        self.states = np.append(self.states, state)

        if self.counter == self.k:
            return True

        self.counter = self.counter + 1

        return False


class ScipyProgressLogger:
    def __init__(self):
        self.states = np.array([])
        self.counter = 0

    def __call__(self, xk) -> bool:
        self.states = np.append(self.states, {"variables": xk, "nit": self.counter})
        self.counter = self.counter + 1

        if self.counter == 4:
            return True

        return False


@pytest.mark.parametrize(
    "solver",
    [
        "steepest-gradient-descent",
        "waypoint-gradient-descent",
        "adaptive-moment-estimation",
    ],
)
def test_solvers_callback(solver):
    iris = datasets.load_iris()

    for k in [0, 1, 2, 4, 10]:
        logger = ProgressLogger(k=k)

        estimator = GLVQ(solver_type=solver, solver_params={"callback": logger})

        pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)

        pipeline.fit(iris.data, iris.target)

        if k == 0:
            assert logger.states[-1]["nit"] == "Initial"
        else:
            assert logger.states[-1]["nit"] == k


@pytest.mark.parametrize("solver", ["lbfgs", "bfgs"])
def test_scipy_solvers_callback(solver):
    iris = datasets.load_iris()

    logger = ScipyProgressLogger()

    estimator = GLVQ(solver_type=solver, solver_params={"callback": logger})

    pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)

    pipeline.fit(iris.data, iris.target)


# def check_init_solver(solver_string):
#     distatance_class = init_class(distances, distance_string)
#
#     assert isinstance(distatance_class, type)
#
#     distance_instance = distatance_class()
#
#     assert isinstance(distance_instance, DistanceBaseClass)
#
#     return distatance_class
#
# def test_aliases():
#     for value in ALIASES.keys():
#         check_init_activation(value)
