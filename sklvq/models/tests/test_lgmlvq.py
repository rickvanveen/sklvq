import pytest

from sklvq.models.lgmlvq import LGMLVQ

from sklearn.datasets import load_iris


def test_lgmlvq():
    iris = load_iris()

    # Localization option ["p", "c"]
    model = LGMLVQ(localization="p")
    model.fit(iris.data, iris.target)

    model_params = model.get_params()
    assert type(model_params) is tuple

    LGMLVQ.normalize_params(model_params)

