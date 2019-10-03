from ..misc import utils

from .activation_base_class import ActivationBaseClass


ALIASES = {'soft+': 'soft-plus'}
BASE_CLASS = ActivationBaseClass
PACKAGE = 'sklvq.activations'


def grab(class_type, class_params):
    if class_type in ALIASES.keys():
        class_type = ALIASES[class_type]

    module_name, class_name = utils.process(class_type)

    return utils.find(PACKAGE, module_name, class_name, class_params, BASE_CLASS)