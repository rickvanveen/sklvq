import pytest
import numpy as np

from sklearn.utils.estimator_checks import check_estimator
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import make_pipeline

from .. import GLVQ
from .. import GMLVQ
from .. import LGMLVQ


@pytest.mark.parametrize("estimator", [GLVQ, GMLVQ, LGMLVQ])
def test_estimators(estimator):
    instance = estimator()
    return check_estimator(instance)