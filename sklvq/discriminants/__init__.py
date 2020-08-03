from .base import DiscriminativeBaseClass
from .relative_distance import RelativeDistance

from typing import Union
from sklvq.misc import utils

__all__ = ["DiscriminativeBaseClass", "RelativeDistance"]

ALIASES = {"reldist": "relative-distance"}
PACKAGE = "sklvq.discriminants"


def grab(
    class_type: Union[str, type],
    class_args: list = None,
    class_kwargs: dict = None,
    whitelist: list = None,
) -> Union[DiscriminativeBaseClass, object]:
    return utils.grab(class_type, class_args, class_kwargs, ALIASES, whitelist, PACKAGE)
