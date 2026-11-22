import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from sklvq.models import GLVQ, GMLVQ, LGMLVQ


@pytest.mark.parametrize("solver", ["sgd", "wgd", "adam", "bfgs", "lbfgs"])
@pytest.mark.parametrize("stop", ["Initial", 2])
def test_solvers_callback(solver, stop):
    """Stopping before or during training must leave the model at the reported parameters."""
    data, labels = load_iris(return_X_y=True)
    states = []

    def callback(state):
        states.append(state)
        return state["nit"] == stop

    limits = {"options": {"maxiter": 12}} if solver in ("bfgs", "lbfgs") else {"max_runs": 12}
    model = GLVQ(solver_type=solver, solver_params={"callback": callback, **limits}, random_state=0)
    model.fit(StandardScaler().fit_transform(data), labels)

    expected_iterations = ["Initial"] if stop == "Initial" else ["Initial", 1, 2]
    assert [state["nit"] for state in states] == expected_iterations
    assert all(state["max_runs"] == 12 for state in states)
    assert states[-1]["model_params"].keys() == {"prototypes"}
    np.testing.assert_array_equal(model.prototypes_, states[-1]["model_params"]["prototypes"])


@pytest.mark.parametrize(("estimator", "solver"), [(LGMLVQ, "wgd"), (GMLVQ, "adam"), (GMLVQ, "lbfgs")])
def test_mutating_snapshots_does_not_change_training(estimator, solver):
    """Overwriting every exposed array must leave later snapshots and the fitted model alone."""
    data, labels = load_iris(return_X_y=True)
    data = StandardScaler().fit_transform(data)
    options = {"options": {"maxiter": 4}} if solver == "lbfgs" else {"max_runs": 4}
    if solver == "wgd":
        options.update(k=2, step_size=[0.1, 0.2])
    control = estimator(random_state=0, solver_type=solver, solver_params=options).fit(data, labels)
    saved = []

    def overwrite(state):
        params = state["model_params"]
        for value in (*params.values(), *state.values()):
            if isinstance(value, np.ndarray):
                value.fill(np.nan)
        saved.append(params)

    model = estimator(random_state=0, solver_type=solver, solver_params={**options, "callback": overwrite})
    model.fit(data, labels)
    np.testing.assert_array_equal(model.get_variables(), control.get_variables())
    assert len(saved) > 1
    for params in saved:
        assert params.keys() == {"prototypes", "omega"}
        for name, value in params.items():
            assert value.shape == getattr(model, f"{name}_").shape
            assert np.isnan(value).all()
    assert not np.shares_memory(saved[0]["omega"], saved[1]["omega"])


def test_scipy_reports_corrected_accepted_iterate():
    """Snapshot, cost and final model must describe the same corrected, normalized SciPy iterate."""
    data = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 4.0]])
    labels = np.array([0, 0, 1, 1])
    states = []

    def stop(state):
        states.append(state)
        return state["nit"] == 1

    model = GMLVQ(
        relevance_correction=np.diag([1.0, 0.5]), random_state=0, solver_type="bfgs", solver_params={"callback": stop}
    ).fit(data, labels)
    accepted = states[-1]
    assert accepted["nit"] == 1
    np.testing.assert_array_equal(accepted["model_params"]["omega"], model.omega_)
    assert accepted["fun"] == pytest.approx(model._objective(model, data, labels))
