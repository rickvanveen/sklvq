from sklearn.utils.estimator_checks import check_estimator
from lvqtoolbox import (TemplateEstimator, TemplateClassifier,
                         TemplateTransformer)
from lvqtoolbox import GeneralizedLearningVectorQuantization


def test_glvq():
	return check_estimator(GeneralizedLearningVectorQuantization)
	

def test_estimator():
    return check_estimator(TemplateEstimator)


def test_classifier():
    return check_estimator(TemplateClassifier)


def test_transformer():
    return check_estimator(TemplateTransformer)
