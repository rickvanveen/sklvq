from .base import DistanceBaseClass

from .euclidean import Euclidean
from .nan_euclidean import NanEuclidean
from .squared_euclidean import SquaredEuclidean
from .squared_nan_euclidean import SquaredNanEuclidean
from .adaptive_squared_euclidean import AdaptiveSquaredEuclidean
from .adaptive_squared_nan_euclidean import AdaptiveSquaredNanEuclidean
from .local_adaptive_squared_euclidean import LocalAdaptiveSquaredEuclidean
from .local_adaptive_squared_nan_euclidean import LocalAdaptiveSquaredNanEuclidean


__all__ = [
    "DistanceBaseClass",
    "Euclidean",
    "NanEuclidean",
    "SquaredEuclidean",
    "SquaredNanEuclidean",
    "AdaptiveSquaredEuclidean",
    "AdaptiveSquaredNanEuclidean",
    "LocalAdaptiveSquaredEuclidean",
    "LocalAdaptiveSquaredNanEuclidean",
]

from ..misc import utils

ALIASES = {"sqeuclidean": "squared-euclidean", "cre": "cumulative-residual-entropy"}
BASE_CLASS = DistanceBaseClass
PACKAGE = "sklvq.distances"


def grab(class_type, class_params):
    return utils.grab(class_type, class_params, ALIASES, PACKAGE, BASE_CLASS)
