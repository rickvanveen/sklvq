def has_call_method(callable_object):
    assert hasattr(callable_object, '__call__')


def has_gradient_method(callable_object):
    assert hasattr(callable_object, 'gradient')
