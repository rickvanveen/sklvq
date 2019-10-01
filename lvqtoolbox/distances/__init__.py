from importlib import import_module

from .distance_base_class import DistanceBaseClass

CLASS_ALIASES = {'sqeuclidean': 'SquaredEuclidean'}
MODULE_ALIASES = {'sqeuclidean': 'squared_euclidean'}

def create(distance_type, distance_params):
    try:
        if '.' in distance_type:
            module_name, class_name = distance_type.rsplit('.', 1)
        else:
            module_name = distance_type
            class_name = distance_type

        if module_name in MODULE_ALIASES.keys():
            module_name = MODULE_ALIASES[module_name]

        distance_module = import_module('.' + module_name, package='lvqtoolbox.distances')

        if class_name in CLASS_ALIASES.keys():
            class_name = CLASS_ALIASES[class_name]
        else:
            class_name = class_name.capitalize()

        distance_class = getattr(distance_module, class_name)

        distance_params = distance_params if distance_params is not None else {}
        instance = distance_class(**distance_params)

    except (AttributeError, ModuleNotFoundError):
        raise ImportError('{} is not part of our distance function collection!'.format(distance_type))
    else:
        if not issubclass(distance_class, DistanceBaseClass):
            raise ImportError(
                "We currently don't have {}, but you are welcome to send in the request for it!".format(
                    distance_class))

    return instance
