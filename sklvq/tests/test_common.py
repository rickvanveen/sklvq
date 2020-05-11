import pytest

from sklearn.utils.estimator_checks import check_estimator

from .. import GLVQClassifier
from .. import GMLVQClassifier
from .. import LGMLVQClassifier


@pytest.mark.parametrize(
    "estimator", [GLVQClassifier, GMLVQClassifier, LGMLVQClassifier]
)
def test_estimators(estimator):
    return check_estimator(estimator)
