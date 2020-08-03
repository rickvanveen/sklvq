from . import ScipyBaseSolver
from ..objectives import ObjectiveBaseClass


class LimitedMemoryBfgs(ScipyBaseSolver):
    """ LimitedMemoryBfgs

    See the documentation of scipy for a complete parameter list and inner workings.

    Parameters
    ----------
    jac: None
        Is set automatically to objective gradient method. However, if no gradient function
        is available, e.g., for a custom distance function, then jac can be set to None.
    callback: callable
        Differently from non-scipy solvers the signature is callback(xk) with xk the current
        set of variables.

    """

    def __init__(self, objective: ObjectiveBaseClass, **kwargs):
        super(LimitedMemoryBfgs, self).__init__(
            objective, method="L-BFGS-B", **kwargs
        )
