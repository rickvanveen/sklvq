from ..misc.utils import find
from .objective_base_class import ObjectiveBaseClass

class_aliases = {'general-objective': 'GeneralizedObjectiveFunction'}
module_aliases = {'general-objective': 'generalized_objective_function'}


# TODO these can me put into a single module

def create(objective_type, objective_params):
    return find(objective_type, objective_params, ObjectiveBaseClass, 'sklvq.objectives',
                class_aliases=class_aliases, module_aliases=module_aliases)
