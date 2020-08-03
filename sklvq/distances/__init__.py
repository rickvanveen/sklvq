from .base import DistanceBaseClass

from .euclidean import Euclidean
from .squared_euclidean import SquaredEuclidean
from .adaptive_squared_euclidean import AdaptiveSquaredEuclidean
from .local_adaptive_squared_euclidean import (
    LocalAdaptiveSquaredEuclidean,
)

from typing import Union
from sklvq.misc import utils

__all__ = [
    "DistanceBaseClass",
    "Euclidean",
    "SquaredEuclidean",
    "AdaptiveSquaredEuclidean",
    "LocalAdaptiveSquaredEuclidean",
]

ALIASES = {"sqeuclidean": "squared-euclidean"}
PACKAGE = "sklvq.distances"


def grab(
    class_type: Union[str, type],
    class_args: list = None,
    class_kwargs: dict = None,
    whitelist: list = None,
) -> Union[DistanceBaseClass, object]:

    return utils.grab(class_type, class_args, class_kwargs, ALIASES, whitelist, PACKAGE)
