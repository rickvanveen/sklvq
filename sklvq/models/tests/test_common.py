import pytest
import numpy as np

from sklearn.utils.estimator_checks import check_estimator
from sklearn import datasets

from .. import GLVQ
from .. import GMLVQ
from .. import LGMLVQ

from sklvq.activations import Identity


@pytest.mark.parametrize("estimator", [GLVQ, GMLVQ, LGMLVQ])
def test_estimators(estimator):
    instance = estimator()
    return check_estimator(instance)


@pytest.mark.parametrize("estimator", [GLVQ, GMLVQ, LGMLVQ])
def test_common_hyper_parameters(estimator):
    X, y = datasets.load_iris(return_X_y=True)

    # Prototype initialization
    with pytest.raises(ValueError):
        estimator(prototype_n_per_class=np.array([1, 1])).fit(X, y)

    with pytest.raises(ValueError):
        estimator(prototype_n_per_class="100").fit(
            np.array([[1, 2, 3], [1, 2, 3]]), np.array([1, 2])
        )

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
        estimator(activation_type=activation_type, activation_params={"beta": 0}).fit(
            X, y
        )


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

    X_hat = estimator().fit(X, y).transform(X, scale=True)
    assert not np.all(np.isnan(X_hat))

    assert pytest.approx(X_hat, estimator().fit_transform(X, y, scale=True))
