import numpy as np
import pytest

from .._adaptive_moment_estimation import AdaptiveMomentEstimation
from sklvq.objectives import GeneralizedLearningObjective


def test_steepest_gradient_descent():

    objective = GeneralizedLearningObjective(
        activation_type="identity",
        activation_params=None,
        discriminant_type="relative-distance",
        discriminant_params=None,
    )

    with pytest.raises(ValueError):
        AdaptiveMomentEstimation(objective, max_runs=-1)

    with pytest.raises(ValueError):
        AdaptiveMomentEstimation(objective, max_runs=0)

    with pytest.raises(ValueError):
        AdaptiveMomentEstimation(objective, beta1=1.1)

    with pytest.raises(ValueError):
        AdaptiveMomentEstimation(objective, beta1=-1)

    with pytest.raises(ValueError):
        AdaptiveMomentEstimation(objective, beta2=1.1)

    with pytest.raises(ValueError):
        AdaptiveMomentEstimation(objective, beta2=-1)

    with pytest.raises(ValueError):
        AdaptiveMomentEstimation(objective, step_size=np.array([1, -1]))

    with pytest.raises(ValueError):
        AdaptiveMomentEstimation(objective, step_size=np.array([1, 0]))

    with pytest.raises(ValueError):
        AdaptiveMomentEstimation(objective, epsilon=-1)

    with pytest.raises(ValueError):
        AdaptiveMomentEstimation(objective, callback=0)
