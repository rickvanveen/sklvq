import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklvq import GLVQClassifier
from sklvq import GMLVQClassifier


@pytest.mark.parametrize(
    "estimator", [GLVQClassifier, GMLVQClassifier]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
