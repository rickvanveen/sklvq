from ._base import ObjectiveBaseClass
from ._generalized_learning_objective import GeneralizedLearningObjective
from sklvq._utils import _import_from_string

__all__ = ["ObjectiveBaseClass", "GeneralizedLearningObjective"]

ALIASES = {}


def import_from_string(class_string, valid_strings=None) -> type:
    return _import_from_string(
        __name__, class_string, ALIASES, "objective_type", valid_strings
    )
