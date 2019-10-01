from importlib import import_module

from .activation_base_class import ActivationBaseClass

CLASS_ALIASES = {'soft+': 'SoftPlus'}
MODULE_ALIASES = {'soft+': 'soft_plus'}


def create(activation_type, activation_params):
    try:
        if '.' in activation_type:
            module_name, class_name = activation_type.rsplit('.', 1)
        else:
            module_name = activation_type
            class_name = activation_type

        if module_name in MODULE_ALIASES.keys():
            module_name = MODULE_ALIASES[module_name]

        activation_module = import_module('.' + module_name, package='lvqtoolbox.activations')

        if class_name in CLASS_ALIASES.keys():
            class_name = CLASS_ALIASES[class_name]
        else:
            class_name = class_name.capitalize()

        activation_class = getattr(activation_module, class_name)

        activation_params = activation_params if activation_params is not None else {}

        try:
            instance = activation_class(**activation_params)
        except TypeError:
            instance = activation_class()

    except (AttributeError, ModuleNotFoundError):
        raise ImportError('{} is not part of our activations function collection!'.format(activation_type))
    else:
        if not issubclass(activation_class, ActivationBaseClass):
            raise ImportError(
                "We currently don't have {}, but you are welcome to send in the request for it!".format(activation_class))

    return instance
