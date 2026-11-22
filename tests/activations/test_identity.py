import numpy as np

from sklvq.activations import Identity


def test_identity():
    """Identity passes each score through and gives each sample unit gradient weight."""
    activation = Identity()
    scores = np.array([-0.75, 0.0, 0.5])

    np.testing.assert_array_equal(activation(scores), [-0.75, 0.0, 0.5])
    np.testing.assert_array_equal(activation.gradient(scores), [1.0, 1.0, 1.0])
