import pytest

from sklvq.models.lgmlvq_classifier import LGMLVQClassifier

from sklearn.datasets import load_iris


def test_lgmlvq():
    iris = load_iris()

    # Localization option ["p", "c"]
    model = LGMLVQClassifier(localization="p")
    model.fit(iris.data, iris.target)

    model_params = model.get_params()
    assert type(model_params) is tuple

    LGMLVQClassifier.normalize_params(model_params)

