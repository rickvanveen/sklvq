import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from sklvq.models import GLVQ


@pytest.mark.parametrize("solver", ["sgd", "wgd", "adam"])
@pytest.mark.parametrize("stop", ["Initial", 2])
def test_solvers_callback(solver, stop):
    """Stopping before or during training must leave the model at the reported parameters."""
    data, labels = load_iris(return_X_y=True)
    states = []

    def callback(state):
        states.append(state)
        return state["nit"] == stop

    model = GLVQ(solver_type=solver, solver_params={"callback": callback}, random_state=0)
    model.fit(StandardScaler().fit_transform(data), labels)

    expected_iterations = ["Initial"] if stop == "Initial" else ["Initial", 1, 2]
    assert [state["nit"] for state in states] == expected_iterations
    np.testing.assert_array_equal(model.get_variables(), states[-1]["variables"])


@pytest.mark.parametrize("solver", ["lbfgs", "bfgs"])
def test_scipy_solvers_callback(solver):
    """The SciPy adapters report flat iterates and leave the fitted model at the last iterate."""
    data, labels = load_iris(return_X_y=True)
    iterates = []
    model = GLVQ(solver_type=solver, solver_params={"callback": iterates.append}, random_state=0)
    model.fit(StandardScaler().fit_transform(data), labels)

    assert iterates
    np.testing.assert_array_equal(model.get_variables(), iterates[-1])
