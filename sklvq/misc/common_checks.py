def has_call_method(callable_object):
    return hasattr(callable_object, "__call__")


def has_gradient_method(callable_object):
    return hasattr(callable_object, "gradient")
