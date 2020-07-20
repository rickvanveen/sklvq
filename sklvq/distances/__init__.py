from .base import DistanceBaseClass

from sklvq.distances.euclidean import Euclidean
from sklvq.distances.squared_euclidean import SquaredEuclidean
from sklvq.distances.adaptive_squared_euclidean import AdaptiveSquaredEuclidean
from sklvq.distances.local_adaptive_squared_euclidean import (
    LocalAdaptiveSquaredEuclidean,
)

from typing import Union
from typing import List

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
