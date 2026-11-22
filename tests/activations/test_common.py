import pytest

from sklvq import activations
from sklvq._utils import init_class
from sklvq.activations import Identity, Sigmoid, SoftPlus, Swish


@pytest.mark.parametrize(
    ("name", "expected"),
    [("identity", Identity), ("sigmoid", Sigmoid), ("soft-plus", SoftPlus), ("soft+", SoftPlus), ("swish", Swish)],
)
def test_activation_name(name, expected):
    """Model configuration must select the requested activation, including the soft+ alias."""
    assert init_class(activations, name) is expected


@pytest.mark.parametrize("activation", [Sigmoid, SoftPlus, Swish])
def test_beta(activation):
    """The default steepness is one; zero and negative values are invalid."""
    assert activation().beta == 1
    for beta in (0, -1):
        with pytest.raises(ValueError, match="beta"):
            activation(beta=beta)
