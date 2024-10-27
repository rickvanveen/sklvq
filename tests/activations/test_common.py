from sklvq import activations
from sklvq._utils import init_class
from sklvq.activations import ALIASES
from sklvq.activations import ActivationBaseClass


def check_init_activation(activation_string):
    activation_class = init_class(activations, activation_string)

    assert isinstance(activation_class, type)

    activation_instance = activation_class()

    assert isinstance(activation_instance, ActivationBaseClass)

    return activation_class


def test_aliases():
    for value in ALIASES.keys():
        check_init_activation(value)
