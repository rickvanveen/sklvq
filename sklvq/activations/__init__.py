""" Package containing activation functions
"""

from .base import ActivationBaseClass
from .identity import Identity
from .sigmoid import Sigmoid
from .soft_plus import SoftPlus
from .swish import Swish

__all__ = ["ActivationBaseClass", "Identity", "Sigmoid", "SoftPlus", "Swish"]

from ..misc import utils

ALIASES = {"soft+": "soft-plus"}
BASE_CLASS = ActivationBaseClass
PACKAGE = "sklvq.activations"


def grab(class_type, class_args=[], class_kwargs={}, whitelist=[]):
    return utils.grab(
        class_type, class_args, class_kwargs, ALIASES, whitelist, PACKAGE, BASE_CLASS
    )
