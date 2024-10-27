from importlib import import_module
from typing import List


def _parse_module_string(parameter_string: str) -> str:
    # Expects all lower case input and converts all hyphens "-" in underscores "_"
    return "_" + parameter_string.replace("-", "_")


def _parse_class_string(parameter_string: str) -> str:
    # Expect all lower case input and converts it to camel-case. The input is split on hyphens "-"
    return "".join(
        [substring.capitalize() for substring in parameter_string.rsplit("-")]
    )


def _parse_import_location_from_string(parameter_string: str) -> (str, str):
    # Converts everything to lower case
    parameter_string = parameter_string.casefold()

    # Return "correct" module and class strings
    return _parse_module_string(parameter_string), _parse_class_string(parameter_string)


def _import_class_from_string(package_string: str, class_string: str) -> type:
    module_string, class_string = _parse_import_location_from_string(class_string)

    try:
        module = import_module("." + module_string, package_string)
        specified_class = getattr(module, class_string)
    except (ModuleNotFoundError, AttributeError, TypeError):
        raise ImportError(
            "{}.{} is not part of our collection".format(module_string, class_string)
        )
    return specified_class


def _import_from_string(
    package,
    class_string: str,
    aliases: dict,
    param_name: str,
    valid_strings: List[str] = None,
) -> type:
    if class_string in aliases.keys():
        class_string = aliases[class_string]

    if valid_strings is not None:
        if not (class_string in valid_strings):
            raise ValueError(f"Provided {param_name} is invalid.")

    return _import_class_from_string(package, class_string)


def init_class(package, class_type, valid_class_types=None):
    if isinstance(class_type, str):
        parameter_class = package.import_from_string(class_type, valid_class_types)
    elif isinstance(class_type, type):
        parameter_class = class_type
    else:
        raise ValueError("Provided value '{}' is invalid.".format(class_type))
    return parameter_class
