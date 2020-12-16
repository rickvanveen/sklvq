from ._base import DiscriminantBaseClass
from ._relative_distance import RelativeDistance
from sklvq._utils import _import_from_string

__all__ = ["DiscriminantBaseClass", "RelativeDistance"]

ALIASES = {}


def import_from_string(class_string, valid_strings=None) -> type:
    return _import_from_string(
        __name__, class_string, ALIASES, "discriminant_type", valid_strings
    )
