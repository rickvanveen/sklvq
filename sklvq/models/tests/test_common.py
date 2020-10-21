import pytest
from sklearn.utils.estimator_checks import check_estimator

from .. import GLVQ
from .. import GMLVQ
from .. import LGMLVQ


@pytest.mark.parametrize("estimator", [GLVQ, GMLVQ, LGMLVQ])
def test_estimators(estimator):
    instance = estimator()
    return check_estimator(instance)
