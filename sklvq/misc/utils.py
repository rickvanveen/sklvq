from importlib import import_module


# TODO: Documentation
# TODO: Extend to include default rules similar to sklearn api
def find(object_type, object_params, base_class, package, class_aliases=None, module_aliases=None):
    try:
        if '.' in object_type:
            module_name, class_name = object_type.rsplit('.', 1)
        else:
            module_name = object_type
            class_name = object_type

        # Lazy evaluation?
        if (module_aliases is not None) and (module_name in module_aliases.keys()):
            module_name = module_aliases[module_name]

        object_module = import_module('.' + module_name, package=package)

        # Lazy evaluation?
        if (class_aliases is not None) and (class_name in class_aliases.keys()):
            class_name = class_aliases[class_name]
        else:
            class_name = class_name.capitalize()

        object_class = getattr(object_module, class_name)

        # In the case None are given but the object does accept arguments
        object_params = object_params if object_params is not None else {}

        # In the case arguments are given but the object doesn't accept any.
        try:
            instance = object_class(**object_params)
        except TypeError:
            instance = object_class()

    except (AttributeError, ModuleNotFoundError):
        raise ImportError('{} is not part of our collection or an alias needs to be created!'.format(object_type))
    else:
        if not issubclass(object_class, base_class):
            raise ImportError(
                "We currently don't have {}, but you are welcome to send in the request for it!".format(object_class))

    return instance
