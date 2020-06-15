""" Package containing activation functions
"""

from sklvq.activations.base import ActivationBaseClass
from sklvq.activations.identity import Identity
from sklvq.activations.sigmoid import Sigmoid
from sklvq.activations.soft_plus import SoftPlus
from sklvq.activations.swish import Swish

from typing import Union

from sklvq.misc import utils

__all__ = ["ActivationBaseClass", "Identity", "Sigmoid", "SoftPlus", "Swish"]

ALIASES = {"soft+": "soft-plus"}
BASE_CLASS = ActivationBaseClass
PACKAGE = "sklvq.activations"


def grab(
    class_type: Union[str, type],
    class_args: list = None,
    class_kwargs: dict = None,
    whitelist: list = None,
) -> Union[ActivationBaseClass, object]:
    return utils.grab(
        class_type, class_args, class_kwargs, ALIASES, whitelist, PACKAGE, BASE_CLASS
    )
