from sklvq._utils import _import_from_string
from sklvq.activations._base import ActivationBaseClass
from sklvq.activations._identity import Identity
from sklvq.activations._sigmoid import Sigmoid
from sklvq.activations._soft_plus import SoftPlus
from sklvq.activations._swish import Swish

__all__ = ["ActivationBaseClass", "Identity", "Sigmoid", "SoftPlus", "Swish"]

ALIASES = {"soft+": "soft-plus"}


def import_from_string(class_string, valid_strings=None) -> type:
    return _import_from_string(__name__, class_string, ALIASES, "activation_type", valid_strings)
