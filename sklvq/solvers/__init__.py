from ..misc.utils import find
from .solver_base_class import SolverBaseClass

class_aliases = {'bgd': 'BatchGradientDescent'}
module_aliases = {'bgd': 'batch_gradient_descent'}


def create(solver_type, solver_params):
    return find(solver_type, solver_params, SolverBaseClass, 'sklvq.solvers',
                class_aliases=class_aliases, module_aliases=module_aliases)
