from importlib import import_module

# TODO: Documentation
# TODO: Look into how to restrict access to certain LVQ classifiers... e.g., not all distance measures are suitable for
#  every classifier
# TODO: look into how to deal with custom objects


def grab(class_type, class_params, aliases, package, base_class):
    if callable(class_type):
        return class_type

    if class_type in aliases.keys():
        class_type = aliases[class_type]

    module_name, class_name = process(class_type)

    return find(package, module_name, class_name, class_params, base_class)


# PACKAGE, module_name, class_name, class_params, BASE_CLASS
def find(package, module_name, class_name, class_params, base_class):
    try:
        object_module = import_module('.' + module_name, package=package)
        object_class = getattr(object_module, class_name)

        try:
            instance = object_class(**class_params)
        except TypeError:
            instance = object_class()

    except (AttributeError, ModuleNotFoundError):
        raise ImportError('{} is not part of our collection or '
                          'an alias needs to be created!'.format(module_name.replace('_', '-')))
    else:
        if not issubclass(object_class, base_class):
            raise ImportError(
                "We currently don't have {}, "
                "but you are welcome to send in the request for it!".format(module_name.replace('_', '-')))

    return instance


def process(object_type_argument):
    # RULE: argument given as parameter to LVQ equals 'squared-euclidean' this will look for the SquaredEuclidean
    # object in the squared_euclidean module in the provided package.

    # Construct default module name
    object_type_argument = object_type_argument.casefold()
    module_name = object_type_argument.replace('-', '_')

    # Construct default class name
    class_name = ''
    object_type_parts = object_type_argument.rsplit('-')
    for part in object_type_parts:
        class_name += part.capitalize()

    return module_name, class_name
