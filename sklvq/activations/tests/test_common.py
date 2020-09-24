from sklvq import activations
from sklvq.activations import ActivationBaseClass
from sklvq._utils import init_class

from sklvq.activations import ALIASES

def check_init_activation(activation_string):
    activation_class = init_class(activations, activation_string)

    assert isinstance(activation_class, type)

    activation_instance = activation_class()

    assert isinstance(activation_instance, ActivationBaseClass)

    return activation_class


def test_aliases():
    for value in ALIASES.values():
        check_init_activation(value)