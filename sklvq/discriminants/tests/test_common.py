from sklvq import discriminants
from sklvq._utils import init_class
from sklvq.discriminants import DiscriminantBaseClass


def check_init_discriminant(discriminant_string):
    discriminant_class = init_class(discriminants, discriminant_string)

    assert isinstance(discriminant_class, type)

    activation_instance = discriminant_class()

    assert isinstance(activation_instance, DiscriminantBaseClass)

    return discriminant_class


# When there are discriminant functions with aliases re-enable
# def test_aliases():
#     for value in ALIASES.keys():
#         check_init_discriminant(value)
