from ._base import DistanceBaseClass
from ._euclidean import Euclidean
from ._squared_euclidean import SquaredEuclidean
from ._adaptive_squared_euclidean import AdaptiveSquaredEuclidean
from ._local_adaptive_squared_euclidean import LocalAdaptiveSquaredEuclidean

from sklvq._utils import _import_from_string

__all__ = [
    "DistanceBaseClass",
    "Euclidean",
    "SquaredEuclidean",
    "AdaptiveSquaredEuclidean",
    "LocalAdaptiveSquaredEuclidean",
]

ALIASES = {}


def import_from_string(class_string, valid_strings=None) -> type:
    return _import_from_string(
        __name__, class_string, ALIASES, "distance_type", valid_strings
    )
