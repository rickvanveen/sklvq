import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.utils import get_tags
from sklearn.utils.estimator_checks import check_estimator

from sklvq import GLVQ, GMLVQ, LGMLVQ
from sklvq.activations import Identity


@pytest.mark.parametrize("estimator", [GLVQ, GMLVQ, LGMLVQ])
def test_estimators(estimator):
    instance = estimator()
    check_estimator(instance)


@pytest.mark.parametrize("estimator", [GLVQ, GMLVQ, LGMLVQ])
def test_nan_validation_and_feature_names(estimator):
    """The NaN-enabled classifier accepts missing observations but still checks its fitted columns."""
    data = pd.DataFrame({"x": [0.0, 0.5, 3.0, 4.0], "y": [np.nan, 1.0, 3.0, 4.0]})
    labels = [0, 0, 1, 1]
    model = estimator(random_state=0)
    assert not get_tags(model).input_tags.allow_nan
    with pytest.raises(ValueError, match="NaN"):
        model.fit(data, labels)

    model.set_params(ensure_all_finite="allow-nan").fit(data, labels)
    assert get_tags(model).input_tags.allow_nan
    assert np.isfinite(model.predict_proba(data)).all()
    with pytest.raises(ValueError, match="feature names"):
        model.predict(data[["y", "x"]])


@pytest.mark.parametrize("estimator", [GLVQ, GMLVQ, LGMLVQ])
def test_common_hyper_parameters(estimator):
    X, y = datasets.load_iris(return_X_y=True)

    # Prototype initialization
    with pytest.raises(ValueError):
        estimator(prototype_n_per_class=np.array([1, 1])).fit(X, y)

    with pytest.raises(TypeError):
        estimator(prototype_n_per_class="100").fit(np.array([[1, 2, 3], [1, 2, 3]]), np.array([1, 2]))

    with pytest.raises(ValueError):
        estimator(prototype_n_per_class=np.array([1, 0, 1])).fit(X, y)

    with pytest.raises(ValueError):
        estimator(prototype_init="abc").fit(X, y)

    m = estimator(prototype_n_per_class=np.array([1, 2, 1])).fit(X, y)
    assert m.prototypes_.shape[0] == 4

    # Activation string which does not exist
    with pytest.raises(ValueError):
        estimator(activation_type="abc123").fit(X, y)

    # Activation object instead of type
    activation_type = Identity()
    with pytest.raises(ValueError):
        estimator(activation_type=activation_type).fit(X, y)

    activation_type = Identity
    with pytest.raises(TypeError):
        estimator(activation_type=activation_type, activation_params={"beta": 0}).fit(X, y)


@pytest.mark.parametrize("estimator", [GMLVQ, LGMLVQ])
def test_shared_memory(estimator):
    X, y = datasets.load_iris(return_X_y=True)
    m = estimator(activation_type="identity").fit(X, y)

    p = m.prototypes_
    m.set_prototypes(np.random.random(size=(3, 4)))
    assert np.shares_memory(p, m.get_prototypes())
    assert np.all(m.get_prototypes() == m.prototypes_)

    o = m.omega_
    m.set_omega(np.random.random(size=m.omega_.shape))
    assert np.shares_memory(o, m.get_omega())
    assert np.all(m.get_omega() == m.omega_)

    model_params = m.to_model_params_view(m.get_variables())
    (p, o) = model_params
    assert np.all(m.prototypes_.shape == p.shape)
    assert np.shares_memory(m.prototypes_, m.get_variables())
    assert np.shares_memory(p, m.get_variables())
    assert np.shares_memory(o, m.get_variables())

    old_variables = m.get_variables()

    p = np.random.random(size=p.shape)
    o = np.random.random(size=o.shape)

    m.set_model_params((p, o))
    assert np.all(m.prototypes_.shape == p.shape)
    assert np.shares_memory(m.prototypes_, m.get_variables())
    assert not np.shares_memory(p, m.get_variables())
    assert not np.shares_memory(o, m.get_variables())

    assert np.shares_memory(old_variables, m.get_variables())


@pytest.mark.parametrize("estimator", [GMLVQ, LGMLVQ])
def test_shared_hyper_params(estimator):
    X, y = datasets.load_iris(return_X_y=True)

    with pytest.raises(ValueError):
        estimator(relevance_normalization="True").fit(X, y)

    with pytest.raises(ValueError):
        estimator(relevance_normalization=6.0).fit(X, y)

    with pytest.raises(ValueError):
        estimator(relevance_n_components="none").fit(X, y)

    with pytest.raises(ValueError):
        estimator(relevance_n_components=120).fit(X, y)

    with pytest.raises(ValueError):
        estimator(relevance_n_components=-120).fit(X, y)

    with pytest.raises(ValueError):
        estimator(relevance_n_components=6.0).fit(X, y)

    with pytest.raises(ValueError):
        estimator(relevance_init="abc").fit(X, y)

    estimator(relevance_init="random").fit(X, y)

    with pytest.raises(ValueError):
        estimator(relevance_init=6).fit(X, y)

    X_hat = estimator(random_state=31415).fit(X, y).transform(X, scale=True)
    assert np.isfinite(X_hat).all()

    X_hat2 = estimator(random_state=31415).fit_transform(X, y, scale=True)
    assert np.array_equiv(X_hat, X_hat2)


@pytest.mark.parametrize("estimator", [GLVQ, GMLVQ, LGMLVQ])
def test_prediction(estimator):
    """Binary thresholds act on the second class probability, not the raw decision score."""
    prototypes = np.array([[0.0], [4.0]])
    model = estimator(solver_params={"callback": lambda state: True}, random_state=0).fit(prototypes, ["a", "z"])
    model.set_prototypes(prototypes)
    X = np.array([[1.0], [2.0], [3.0]])

    # Squared distances to a and z are (1,9), (4,4), and (9,1).
    scores = model.decision_function(X)
    assert scores.shape == (3,)
    np.testing.assert_allclose(scores, [-0.8, 0.0, 0.8])
    np.testing.assert_allclose(model.predict_proba(X), [[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    np.testing.assert_array_equal(model.predict(X), ["a", "a", "z"])
    np.testing.assert_array_equal(model.predict(X, threshold=0.2), ["a", "z", "z"])

    with pytest.raises(ValueError):
        model.predict(X, threshold=-1)

    with pytest.raises(ValueError):
        model.predict(X, threshold=2.0)

    with pytest.raises(TypeError):
        model.predict(X, threshold="0.5")


def test_multiclass_prediction_uses_nearest_prototype_per_class():
    """Class a's second prototype can win; probabilities follow classes_, not prototype order."""
    prototypes = np.array([[0.0], [8.0], [4.0], [12.0]])
    model = GLVQ(prototype_n_per_class=[2, 1, 1], solver_params={"callback": lambda state: True}, random_state=0)
    model.fit(prototypes, ["a", "a", "b", "c"])
    model.set_prototypes(prototypes)
    data = np.array([[7.0], [11.0]])

    # Nearest squared distances by class: (1,9,25) and (9,49,1).
    expected_scores = np.array([[0.8, -0.8, -24 / 26], [-0.8, -48 / 50, 0.8]])
    scores = model.decision_function(data)
    assert scores.shape == (2, 3)
    np.testing.assert_allclose(scores, expected_scores)
    weights = np.exp(expected_scores)
    np.testing.assert_allclose(model.predict_proba(data), weights / weights.sum(axis=1, keepdims=True))
    np.testing.assert_array_equal(model.predict(data), ["a", "c"])


@pytest.mark.parametrize("estimator", [GMLVQ, LGMLVQ])
def test_normalize_inactive_relevance_gradient(estimator):
    """A zero relevance update must stay zero when waypoint descent normalizes its gradient."""
    data = np.array([[0.0, 0.0], [2.0, 1.0]])
    model = estimator(solver_params={"callback": lambda state: True}, random_state=0).fit(data, [0, 1])
    gradient = np.zeros_like(model.get_variables())
    prototypes, omega = model.to_model_params_view(gradient)
    prototypes[0] = [3.0, 4.0]

    expected = np.zeros_like(omega)
    if estimator is LGMLVQ:
        # An inactive local matrix must not prevent its neighbor from being normalized.
        omega[1, 0] = [3.0, 4.0]
        expected[1, 0] = [0.6, 0.8]

    model.normalize_variables(gradient)

    np.testing.assert_allclose(omega, expected)
