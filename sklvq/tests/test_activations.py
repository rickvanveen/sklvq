from .. import activations

import nose.tools as nt


def test_identity_init():
    identity_instance = activations.grab('identity', None)

    nt.eq_(identity_instance(1), 1)
    nt.eq_(identity_instance.gradient(42), 1)


def test_sigmoid_init():
    sigmoid_instance = activations.grab('sigmoid', {'beta': 6})
    nt.eq_(sigmoid_instance.beta, 6)

    sigmoid_instance = activations.grab('sigmoid', None)
    nt.eq_(sigmoid_instance.beta, 1)
