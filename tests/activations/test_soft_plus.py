import numpy as np

from .test_common import check_init_activation

import pytest


def test_soft_plus():
    soft_plus_class = check_init_activation("soft-plus")

    assert isinstance(soft_plus_class, type)

    soft_plus = soft_plus_class(beta=2)

    assert soft_plus.beta == 2

    soft_plus = soft_plus_class()

    assert soft_plus.beta == 1

    wrong_betas = [-1, 0]
    for wrong_beta in wrong_betas:
        with pytest.raises(ValueError):
            soft_plus_class(beta=wrong_beta)

    # >0 at x = 0
    assert soft_plus(np.array([0])) > 0

    # >0 at x < 0
    assert np.all(soft_plus(np.array([-2, -4, -6, -8, -20, -50, -100]) > 0))

    # Increasing upto gradient of beta...
    assert np.max(soft_plus.gradient(np.array([0, 4, 8, 12, 20]))) == pytest.approx(
        soft_plus.beta
    )

    soft_plus = soft_plus_class(beta=10)

    # Increasing upto gradient of beta...
    assert np.max(soft_plus.gradient(np.array([0, 4, 8, 12, 20]))) == pytest.approx(
        soft_plus.beta
    )

    # soft plus gradient becomes nan when 1 + (beta * e^x) becomes inf. Which happens with larger x.
    # which should probably raise an error.

    # Potentially raising errors when division by 0.0 or inf happens....
