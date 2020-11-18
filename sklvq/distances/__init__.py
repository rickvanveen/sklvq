from ._base import DistanceBaseClass
from ._euclidean import Euclidean
from ._squared_euclidean import SquaredEuclidean
from ._adaptive_squared_euclidean import AdaptiveSquaredEuclidean
from ._local_adaptive_squared_euclidean import LocalAdaptiveSquaredEuclidean

from sklvq._utils import _import_class_from_string

__all__ = [
    "DistanceBaseClass",
    "Euclidean",
    "SquaredEuclidean",
    "AdaptiveSquaredEuclidean",
    "LocalAdaptiveSquaredEuclidean",
]

ALIASES = {}


def import_from_string(class_string, valid_strings=None) -> type:
    # if class_string in ALIASES.keys():
    #     class_string = ALIASES[class_string]

    if valid_strings is not None:
        if not (class_string in valid_strings):
            raise ValueError("Provided distance_type is invalid.")

    return _import_class_from_string(__name__, class_string)
