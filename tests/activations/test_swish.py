import numpy as np

from .test_common import check_init_activation

import pytest


def test_swish():
    swish_class = check_init_activation("swish")

    assert isinstance(swish_class, type)

    swish = swish_class(beta=2)

    assert swish.beta == 2

    swish = swish_class()

    assert swish.beta == 1

    wrong_betas = [-1, 0]
    for wrong_beta in wrong_betas:
        with pytest.raises(ValueError):
            swish_class(beta=wrong_beta)

    # x = 0 should be 0
    assert swish(np.array([0])) == 0
    # x < 0 is negative and goes to 0
    assert swish(np.array([-1])) < 0

    # Swish is approximately the identity function, when x gets large enough, thus gradient is 1.
    x = np.array([4, 8, 12, 20, 50, 100, 200])
    assert swish(x) == pytest.approx(x, abs=1e-1)
    assert swish.gradient(x) == pytest.approx(1, abs=1e-1)

    swish = swish_class(beta=10)

    # x = 0 should still be 0
    assert swish(np.array([0])) == 0
    # x < 0 should still be negative and goes to 0
    assert swish(np.array([-1])) < 0

    # Swish is with higher beta less approximately the identity function, when x gets large
    # enough, thus gradient is 1.
    assert swish(np.array([4, 8, 12, 20, 50, 100, 200])) == pytest.approx(
        np.array([4, 8, 12, 20, 50, 100, 200]), abs=1e-10
    )
    assert swish.gradient(x) == pytest.approx(1, abs=1e-10)

    # Potentially raising errors when division by 0.0/inf/nans happens....
