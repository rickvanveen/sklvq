import numpy as np
import pytest

from .._waypoint_gradient_descent import WaypointGradientDescent
from sklvq.objectives import GeneralizedLearningObjective


def test_steepest_gradient_descent():

    objective = GeneralizedLearningObjective(
        activation_type="identity",
        activation_params=None,
        discriminant_type="relative-distance",
        discriminant_params=None,
    )

    with pytest.raises(ValueError):
        WaypointGradientDescent(objective, max_runs=-1)

    with pytest.raises(ValueError):
        WaypointGradientDescent(objective, max_runs=0)

    with pytest.raises(ValueError):
        WaypointGradientDescent(objective, step_size=-1)

    with pytest.raises(ValueError):
        WaypointGradientDescent(objective, step_size=np.array([1, -1]))

    with pytest.raises(ValueError):
        WaypointGradientDescent(objective, step_size=np.array([1, 0]))

    with pytest.raises(ValueError):
        WaypointGradientDescent(objective, callback=0)

    with pytest.raises(ValueError):
        WaypointGradientDescent(objective, loss=-1)

    with pytest.raises(ValueError):
        WaypointGradientDescent(objective, loss=1.1)

    with pytest.raises(ValueError):
        WaypointGradientDescent(objective, gain=0.9)

    with pytest.raises(ValueError):
        WaypointGradientDescent(objective, k=-1)
