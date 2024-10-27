from sklvq._utils import _import_from_string
from sklvq.solvers._adaptive_moment_estimation import AdaptiveMomentEstimation
from sklvq.solvers._base import ScipyBaseSolver, SolverBaseClass
from sklvq.solvers._broyden_fletcher_goldfarb_shanno import BroydenFletcherGoldfarbShanno
from sklvq.solvers._limited_memory_bfgs import LimitedMemoryBfgs
from sklvq.solvers._steepest_gradient_descent import SteepestGradientDescent
from sklvq.solvers._waypoint_gradient_descent import WaypointGradientDescent

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
    "wgd": "waypoint-gradient-descent",
    "bfgs": "broyden-fletcher-goldfarb-shanno",
    "lbfgs": "limited-memory-bfgs",
    "adam": "adaptive-moment-estimation",
}


def import_from_string(class_string, valid_strings=None) -> type:
    return _import_from_string(
        __name__, class_string, ALIASES, "objective_type", valid_strings
    )
