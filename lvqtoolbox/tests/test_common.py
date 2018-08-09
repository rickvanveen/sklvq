from sklearn.utils.estimator_checks import check_estimator


def test_glvq():
    return check_estimator(GLVQClassifier)
