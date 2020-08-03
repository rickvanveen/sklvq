""" Package containing activation functions
"""

from .base import ActivationBaseClass
from .identity import Identity
from .sigmoid import Sigmoid
from .soft_plus import SoftPlus
from .swish import Swish

from typing import Union

from sklvq.misc import utils

__all__ = ["ActivationBaseClass", "Identity", "Sigmoid", "SoftPlus", "Swish"]

ALIASES = {"soft+": "soft-plus"}
PACKAGE = "sklvq.activations"


def grab(
    class_type: Union[str, type],
    class_args: list = None,
    class_kwargs: dict = None,
    whitelist: list = None,
) -> Union[ActivationBaseClass, object]:
    return utils.grab(class_type, class_args, class_kwargs, ALIASES, whitelist, PACKAGE)
