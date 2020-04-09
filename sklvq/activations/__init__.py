""" Package containing activation functions
"""

from .base import ActivationBaseClass
from .identity import Identity
from .sigmoid import Sigmoid
from .soft_plus import SoftPlus
from .swish import Swish

# TODO: Doest the base class need to be public?
__all__ = ["ActivationBaseClass", "Identity", "Sigmoid", "SoftPlus", "Swish"]

from ..misc import utils

ALIASES = {"soft+": "soft-plus"}
BASE_CLASS = ActivationBaseClass
PACKAGE = "sklvq.activations"


def grab(class_type, class_params):
    return utils.grab(class_type, class_params, ALIASES, PACKAGE, BASE_CLASS)
