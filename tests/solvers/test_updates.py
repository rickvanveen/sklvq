import numpy as np
import pytest

from sklvq import GLVQ, GMLVQ, LGMLVQ
from sklvq.solvers import AdaptiveMomentEstimation, LimitedMemoryBfgs, SteepestGradientDescent, WaypointGradientDescent
from tests.distances._utils import finite_difference


class Quadratic:
    """Each sample contributes w²/2, so expectations do not depend on shuffle order."""

    def __call__(self, model, data, labels):
        return labels.size * np.sum(model.get_variables() ** 2) / 2

    def gradient(self, model, data, labels):
        return labels.size * model.get_variables()


def scalar_model(start):
    model = GLVQ()
    model._variables = np.array([float(start)])
    model._prototypes_shape = (1, 1)
    model.random_state_ = np.random.RandomState(0)
    return model


@pytest.mark.parametrize(("batch_size", "expected"), [(0, 0.5), (2, 5 / 6)])
def test_steepest_descent_updates(batch_size, expected):
    """Apply rates 1/4 and 1/6 over two epochs; the last mini-batch has only one sample."""
    model = scalar_model(4)
    solver = SteepestGradientDescent(Quadratic(), max_runs=2, batch_size=batch_size, step_size=0.25)

    solver.solve(np.zeros((3, 1)), np.zeros(3), model)

    # Full batch: 4 → 1 → 1/2. Mini-batches: 4 → 2 → 3/2 → 1 → 5/6.
    np.testing.assert_allclose(model.get_variables(), [expected])


def test_adam_bias_corrected_steps():
    """The second update uses both samples' gradients, not just the latest gradient."""
    model = scalar_model(2)
    solver = AdaptiveMomentEstimation(Quadratic(), max_runs=1, beta1=0.5, beta2=0.5, step_size=1, epsilon=1)

    solver.solve(np.zeros((2, 1)), np.zeros(2), model)

    # Gradients 2 and 4/3 give second-step corrected moments 14/9 and 68/27.
    expected = 4 / 3 - (14 / 9) / (np.sqrt(68 / 27) + 1)
    np.testing.assert_allclose(model.get_variables(), [expected])


@pytest.mark.parametrize(("start", "expected", "next_rate"), [(4, 1, 1.5), (1.75, 0.25, 0.5)])
def test_waypoint_accepts_lower_cost(start, expected, next_rate):
    """After warm-up, accept either the descent step (gain) or the cheaper waypoint average (loss)."""
    model = scalar_model(start)
    states = []
    solver = WaypointGradientDescent(
        Quadratic(), max_runs=3, k=2, step_size=1, gain=1.5, loss=0.5, callback=states.append
    )

    solver.solve(np.zeros((1, 1)), np.zeros(1), model)

    # From 4 the candidates are 1 and 2.5; from 1.75 they are 0.75 and 0.25.
    np.testing.assert_allclose(model.get_variables(), [expected])
    assert states[-1]["fun"] == pytest.approx(expected**2 / 2)
    assert states[-1]["step_size"] == pytest.approx(next_rate)


@pytest.mark.parametrize("parameter", ["beta1", "beta2"])
def test_adam_rejects_unit_decay(parameter):
    """A decay of one makes the bias-correction denominator zero on the first update."""
    with pytest.raises(ValueError, match=parameter):
        AdaptiveMomentEstimation(Quadratic(), **{parameter: 1.0})


@pytest.mark.parametrize("model_class", [GMLVQ, LGMLVQ])
def test_waypoint_scalar_rate_matches_equal_group_rates(model_class):
    """A shared rate must scale both prototypes and relevance factors even on the first update."""
    data = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 4.0]])
    labels = np.array([0, 0, 1, 1])

    def first_step(rate):
        params = {"step_size": rate, "callback": lambda state: state["nit"] == 1}
        return model_class(solver_type="wgd", solver_params=params, random_state=0).fit(data, labels).get_variables()

    np.testing.assert_allclose(first_step(0.1), first_step([0.1, 0.1]))


@pytest.mark.parametrize(
    "model",
    [
        GMLVQ(),
        LGMLVQ(),
        GMLVQ(relevance_correction=np.array([[1, 0.2], [0, 0.5]])),
        GMLVQ(relevance_correction=np.array([[1, 0.2], [0, 0.5]]), relevance_normalization=False),
    ],
    ids=["global", "local", "corrected-normalized", "corrected-only"],
)
def test_scipy_gradient_at_unnormalized_trial_point(model):
    """SciPy differentiates raw trial parameters; the cost wrapper normalizes their relevance matrix."""
    data = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 4.0]])
    labels = np.array([0, 0, 1, 1])
    model.set_params(random_state=0, solver_params={"callback": lambda state: True}).fit(data, labels)
    solver = LimitedMemoryBfgs(model._objective)
    trial = model.get_variables().copy()
    model.to_omega(trial)[:] *= 3

    expected = finite_difference(lambda: solver._objective_wrapper(trial, model, data, labels), [trial])[0]
    actual = solver._objective_gradient_wrapper(trial, model, data, labels)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
