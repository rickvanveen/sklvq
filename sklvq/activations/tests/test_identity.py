from .test_common import check_init_activation

import numpy as np


# Check if grab returns correct class, defaults of init (if any), basic workings are correct.
def test_identity():
    identity_class = check_init_activation("identity")

    identity = identity_class()

    rng = np.random.RandomState(0)
    x = rng.random_sample((5, 4))

    assert np.all(identity(x) == x)

    assert np.all(identity.gradient(x) == np.ones(x.shape))
