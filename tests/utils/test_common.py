import pytest
from sklvq._utils import _import_class_from_string


def test_utils():
    _import_class_from_string("sklvq.distances", "squared-euclidean")

    # Wrong package
    with pytest.raises(ImportError):
        _import_class_from_string("sklvq.activations", "squared-euclidean")

    # Wrong class
    with pytest.raises(ImportError):
        _import_class_from_string("sklvq.distances", "sigmoid")

    # Case where package/module exists but the class inside the module
    # does not have the same name.
