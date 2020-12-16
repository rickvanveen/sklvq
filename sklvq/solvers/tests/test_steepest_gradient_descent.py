import numpy as np
import pytest

from .._steepest_gradient_descent import SteepestGradientDescent
from sklvq.objectives import GeneralizedLearningObjective
from sklvq.models import GLVQ
from sklearn.datasets import load_iris


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
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError):
        SteepestGradientDescent(objective, batch_size=100000).solve(X, y, model)
