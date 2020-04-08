from .base import DiscriminativeBaseClass
from .relative_distance import RelativeDistance

__all__ = ["DiscriminativeBaseClass", "RelativeDistance"]

from ..misc import utils

ALIASES = {"reldist": "relative-distance"}
BASE_CLASS = DiscriminativeBaseClass
PACKAGE = "sklvq.discriminants"


def grab(class_type, class_params):
    return utils.grab(class_type, class_params, ALIASES, PACKAGE, BASE_CLASS)
