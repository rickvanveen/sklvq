from sklearn.utils.estimator_checks import check_estimator

from lvqtoolbox.v2.models import GLVQClassifier


def test_glvq():
    return check_estimator(GLVQClassifier)