from .base import DistanceBaseClass

# from .adaptive_euclidean import
from .adaptive_squared_euclidean import AdaptiveSquaredEuclidean
from .euclidean import Euclidean
from .squared_euclidean import SquaredEuclidean
from .cosine_distance import CosineDistance

__all__ = [
    "DistanceBaseClass",
    "Euclidean",
    "SquaredEuclidean",
    "AdaptiveSquaredEuclidean",
]

from ..misc import utils

ALIASES = {"sqeuclidean": "squared-euclidean", "cre": "cumulative-residual-entropy"}
BASE_CLASS = DistanceBaseClass
PACKAGE = "sklvq.distances"


def grab(class_type, class_params):
    return utils.grab(class_type, class_params, ALIASES, PACKAGE, BASE_CLASS)
