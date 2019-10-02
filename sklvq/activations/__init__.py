from ..misc.utils import find

from .activation_base_class import ActivationBaseClass

class_aliases = {'soft+': 'SoftPlus'}
module_aliases = {'soft+': 'soft_plus'}


def create(activation_type, activation_params):
    return find(activation_type, activation_params, ActivationBaseClass, 'sklvq.activations',
                class_aliases=class_aliases, module_aliases=module_aliases)