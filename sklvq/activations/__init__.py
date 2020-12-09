from ._base import ActivationBaseClass
from ._identity import Identity
from ._sigmoid import Sigmoid
from ._soft_plus import SoftPlus
from ._swish import Swish

from sklvq._utils import _import_from_string

__all__ = ["ActivationBaseClass", "Identity", "Sigmoid", "SoftPlus", "Swish"]

ALIASES = {"soft+": "soft-plus"}


def import_from_string(class_string, valid_strings=None) -> type:
    return _import_from_string(
        __name__, class_string, ALIASES, "activation_type", valid_strings
    )
