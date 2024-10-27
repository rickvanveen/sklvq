import numpy as np

from .test_common import check_init_activation

import pytest


def test_sigmoid():
    sigmoid_class = check_init_activation("sigmoid")

    sigmoid = sigmoid_class(beta=2)

    assert sigmoid.beta == 2

    sigmoid = sigmoid_class()

    assert sigmoid.beta == 1

    wrong_betas = [-1, 0]
    for wrong_beta in wrong_betas:
        with pytest.raises(ValueError):
            sigmoid_class(beta=wrong_beta)

    # Check if x = 0 equals approximately 0.5
    assert sigmoid(np.array([0])) == pytest.approx(0.5)
    # Check if at x = 0 the gradient equals approximately 0.25
    assert sigmoid.gradient(np.array([0])) == pytest.approx(0.25)

    # "Always" positive with negative numbers
    assert np.all(sigmoid(np.array([-1, -10, -100, -1000, -10000]) > 0))
    # "Always" positive with positive numbers
    assert np.all(sigmoid(np.array([1, 10, 100, 1000, 10000]) > 0))

    # Symmetry
    assert np.all(
        pytest.approx(sigmoid(np.array([1, 2, 3, 4])))
        == (1 - sigmoid(np.array([-1, -2, -3, -4])))
    )

    sigmoid = sigmoid_class(beta=10)

    # Check if x = 0 is not changed
    assert sigmoid(np.array([0])) == pytest.approx(0.5)
    # Check if x = 0 is changed based on beta.
    assert sigmoid.gradient(np.array([0])) == pytest.approx(2.5)

    # Potentially raising errors when division by 0.0 or inf happens....
