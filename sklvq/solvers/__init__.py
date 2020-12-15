from ._base import SolverBaseClass, ScipyBaseSolver
from ._steepest_gradient_descent import SteepestGradientDescent
from ._waypoint_gradient_descent import WaypointGradientDescent
from ._adaptive_moment_estimation import AdaptiveMomentEstimation
from ._limited_memory_bfgs import LimitedMemoryBfgs
from ._broyden_fletcher_goldfarb_shanno import BroydenFletcherGoldfarbShanno

from sklvq._utils import _import_from_string

__all__ = [
    "SolverBaseClass",
    "ScipyBaseSolver",
    "SteepestGradientDescent",
    "WaypointGradientDescent",
    "AdaptiveMomentEstimation",
    "LimitedMemoryBfgs",
    "BroydenFletcherGoldfarbShanno",
]

ALIASES = {
    "sgd": "steepest-gradient-descent",
    "bgd": "steepest-gradient-descent",
    "bfgs": "broyden-fletcher-goldfarb-shanno",
    "lbfgs": "limited-memory-bfgs",
    "adam": "adaptive-moment-estimation",
}


def import_from_string(class_string, valid_strings=None) -> type:
    return _import_from_string(
        __name__, class_string, ALIASES, "objective_type", valid_strings
    )
