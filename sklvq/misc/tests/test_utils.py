from sklvq.misc.utils import parse_class_type, find_and_init, grab
from sklvq import activations, distances
import pytest


def test_parse_class_type():
    module_name, class_name = parse_class_type("squared-euclidean")

    assert module_name == "squared_euclidean"
    assert class_name == "SquaredEuclidean"

    module_name, class_name = parse_class_type("euclidean")

    assert module_name == "euclidean"
    assert class_name == "Euclidean"


def test_find_and_init():
    package = "sklvq.activations"
    module_name = "sigmoid"
    class_name = "Sigmoid"
    class_args = []
    class_kwargs = {}

    object = find_and_init(package, module_name, class_name, class_args, class_kwargs)

    assert isinstance(object, activations.sigmoid.Sigmoid)

    # Providing args without kwargs is not a problem
    class_args = [3]
    object = find_and_init(package, module_name, class_name, class_args, class_kwargs)
    # As long beta is initiated to the correct value
    assert object.beta == 3

    # Only 1 arg while Sigmoid doesn't accept any... with the correct kwarg should raise an
    # exception.
    class_args = [3]
    class_kwargs = {"beta": 6}
    with pytest.raises(ValueError):
        find_and_init(package, module_name, class_name, class_args, class_kwargs)

    # Wrong kwargs are given should throw an exception
    class_args = []
    class_kwargs = {"alpha": 3}
    with pytest.raises(ValueError):
        find_and_init(package, module_name, class_name, class_args, class_kwargs)

    # Correct args/kwargs should change the value.
    class_args = []
    class_kwargs = {"beta": 6}
    object = find_and_init(package, module_name, class_name, class_args, class_kwargs)
    assert object.beta == 6

    # Wrong package
    package = "sklvq.activation"
    module_name = "sigmoid"
    class_name = "Sigmoid"

    with pytest.raises(ImportError):
        find_and_init(package, module_name, class_name, class_args, class_kwargs)

    package = "sklvq.activations"
    module_name = "sigmoi"

    with pytest.raises(ImportError):
        find_and_init(package, module_name, class_name, class_args, class_kwargs)

    package = "sklvq.activations"
    module_name = "sigmoid"
    class_name = "Sigmid"

    with pytest.raises(ImportError):
        find_and_init(package, module_name, class_name, class_args, class_kwargs)


def test_grab():
    class_type = 123
    class_args = []
    class_kwargs = {}
    aliases = {}
    whitelist = []
    package = "sklvq.distances"

    with pytest.raises(ValueError):
        grab(class_type, class_args, class_kwargs, aliases, whitelist, package)

    class_type = "squared-euclidean"
    class_args = None
    class_kwargs = None
    aliases = None
    whitelist = None

    object_instance = grab(class_type, class_args, class_kwargs, aliases, whitelist, package)
    assert isinstance(object_instance, distances.squared_euclidean.SquaredEuclidean)

    whitelist = ["euclidean"]
    with pytest.raises(ValueError):
        grab(class_type, class_args, class_kwargs, aliases, whitelist, package)

    whitelist = ["squared-euclidean"]
    object_instance = grab(class_type, class_args, class_kwargs, aliases, whitelist, package)
    assert isinstance(object_instance, distances.squared_euclidean.SquaredEuclidean)

    class Mockclass:
        def __init__(self, mock_parameter=1):
            self.mock_parameter = mock_parameter

    object_instance = grab(Mockclass, class_args, class_kwargs, aliases, whitelist, package)
    assert isinstance(object_instance, Mockclass)
    assert object_instance.mock_parameter == 1

    class_kwargs = {"mock_parameter": 6}
    object_instance = grab(Mockclass, class_args, class_kwargs, aliases, whitelist, package)
    assert object_instance.mock_parameter == 6

    class_args = [10]
    class_kwargs = {"alpha": 6}
    with pytest.raises(ValueError):
        grab(class_type, class_args, class_kwargs, aliases, whitelist, package)

    # check if aliases work
    class_type = "soft+"
    class_args = None
    class_kwargs = None
    aliases = activations.ALIASES
    whitelist = None
    package = "sklvq.activations"

    object_instance = grab(class_type, class_args, class_kwargs, aliases, whitelist, package)
    assert isinstance(object_instance, activations.soft_plus.SoftPlus)
