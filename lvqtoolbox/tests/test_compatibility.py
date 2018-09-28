from sklearn.utils.estimator_checks import check_estimator

from lvqtoolbox.models import GLVQClassifier, GMLVQClassifier


def test_glvq():
    return check_estimator(GLVQClassifier)

def test_gmlvq():
    return check_estimator(GMLVQClassifier)