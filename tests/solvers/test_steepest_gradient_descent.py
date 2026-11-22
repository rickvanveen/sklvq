import numpy as np
import pytest

from sklvq.models import GLVQ
from sklvq.objectives import GeneralizedLearningObjective
from sklvq.solvers._steepest_gradient_descent import SteepestGradientDescent


def test_steepest_gradient_descent():
    objective = GeneralizedLearningObjective(
        activation_type="identity",
        activation_params=None,
        discriminant_type="relative-distance",
        discriminant_params=None,
    )

    with pytest.raises(ValueError):
        SteepestGradientDescent(objective, max_runs=-1)

    with pytest.raises(ValueError):
        SteepestGradientDescent(objective, max_runs=0)

    with pytest.raises(ValueError):
        SteepestGradientDescent(objective, batch_size=-1)

    with pytest.raises(ValueError):
        SteepestGradientDescent(objective, step_size=-1)

    with pytest.raises(ValueError):
        SteepestGradientDescent(objective, step_size=np.array([1, -1]))

    with pytest.raises(ValueError):
        SteepestGradientDescent(objective, step_size=np.array([1, 0]))

    with pytest.raises(ValueError):
        SteepestGradientDescent(objective, callback=0)

    model = GLVQ()
    data, labels = np.zeros((2, 1)), np.zeros(2)
    with pytest.raises(ValueError):
        SteepestGradientDescent(objective, batch_size=3).solve(data, labels, model)
