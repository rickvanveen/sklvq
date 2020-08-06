from importlib import import_module
from inspect import isclass
from typing import Tuple, Union


def _check_input(class_args, class_kwargs, aliases, whitelist):
    if class_args is None:
        class_args = []

    if class_kwargs is None:
        class_kwargs = {}

    if aliases is None:
        aliases = {}

    if whitelist is None:
        whitelist = []

    return class_args, class_kwargs, aliases, whitelist


def grab(
    class_type: Union[type, str], class_args, class_kwargs, aliases, whitelist, package,
):
    if not (isinstance(class_type, str) or isclass(class_type)):
        raise ValueError("Type parameter should either be a class or string")

    # Set args and kwargs to empty iterables if they are None
    class_args, class_kwargs, aliases, whitelist = _check_input(
        class_args, class_kwargs, aliases, whitelist
    )

    # Check if class_type is actually a custom class
    if not isinstance(class_type, str):
        # If this fails the wrong args and or kwargs were provided. If None, i.e.,
        # empty iterables are provided this should just work.
        try:
            return class_type(*class_args, **class_kwargs)
        except TypeError:
            raise ValueError(
                "{} does not accept the provided arguments.".format(class_type)
            )

    # If an alias is used (this must be a key in the aliases constant of the package) get the
    # actual name of the callable.
    if class_type in aliases.keys():
        class_type = aliases[class_type]

    # When grab is called a list of 'compatible' methods needs to be provided, aka the whiteliste
    if whitelist:
        if class_type not in whitelist:
            raise ValueError(
                "Provided type parameters value is not compatible with this model or does not "
                "exist."
            )

    module_name, class_name = parse_class_type(class_type)

    return find_and_init(package, module_name, class_name, class_args, class_kwargs)


def find_and_init(package, module_name, class_name, class_args, class_kwargs):
    try:
        module = import_module("." + module_name, package=package)
        path_to_class = getattr(module, class_name)
    except (AttributeError, ModuleNotFoundError):
        raise ImportError(
            "{} is not part of our collection or ".format(module_name.replace("_", "-"))
        )

    try:
        return path_to_class(*class_args, **class_kwargs)
    except TypeError:
        raise ValueError(
            "{} does not accept the provided arguments".format(path_to_class)
        )


def parse_class_type(class_type: str) -> Tuple[str, str]:
    #     The string "squared-euclidean" will result in module_name "squared_euclidean" and
    #     class_name "SquaredEuclidean"

    # Convert to lowercase
    class_type = class_type.casefold()

    # Transform - in _ to get the module name
    module_name = class_type.replace("-", "_")

    class_name = ""
    # Split the class_type string based on "-"
    class_type_parts = class_type.rsplit("-")

    # Concat every part but capitalize the first letter
    for class_type_part in class_type_parts:
        class_name += class_type_part.capitalize()

    return module_name, class_name
