from sklearn.utils.estimator_checks import check_estimator

from lvqtoolbox import GLVQ


def test_glvq():
    return check_estimator(GLVQ)
