from sklearn.utils.estimator_checks import check_estimator

from lvqtoolbox import GLVQClassifier

def test_glvq():
    return check_estimator(GLVQClassifier)
