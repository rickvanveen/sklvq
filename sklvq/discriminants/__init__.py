from .base import DiscriminativeBaseClass
from .relative_distance import RelativeDistance

__all__ = ["DiscriminativeBaseClass", "RelativeDistance"]

from ..misc import utils

ALIASES = {"reldist": "relative-distance"}
BASE_CLASS = DiscriminativeBaseClass
PACKAGE = "sklvq.discriminants"


def grab(class_type, class_args=[], class_kwargs={}, whitelist=[]):
    return utils.grab(
        class_type, class_args, class_kwargs, ALIASES, whitelist, PACKAGE, BASE_CLASS
    )
