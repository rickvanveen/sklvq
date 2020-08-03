import pytest
import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

from sklvq.models import GLVQ


class ProgressLogger:
    def __init__(self):
        self.states = np.array([])
        self.counter = 0

    def __call__(self, model, state) -> bool:
        self.states = np.append(self.states, state)
        self.counter = self.counter + 1

        if self.counter == 4:
            return True

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
    "solver", [
        "steepest-gradient-descent",
        "waypoint-gradient-descent",
        "adaptive-moment-estimation",
    ]
)
def test_solvers_callback(solver):
    iris = datasets.load_iris()

    logger = ProgressLogger()

    estimator = GLVQ(solver_type=solver, solver_params={"callback": logger})

    pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)

    pipeline.fit(iris.data, iris.target)

    assert logger.states[-1]["nit"] == 3


@pytest.mark.parametrize(
    "solver", [
        "lbfgs",
        "bfgs",
    ]
)
def test_solvers_callback(solver):
    iris = datasets.load_iris()

    logger = ScipyProgressLogger()

    estimator = GLVQ(solver_type=solver, solver_params={"callback": logger})

    pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)

    pipeline.fit(iris.data, iris.target)
