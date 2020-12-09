from . import ScipyBaseSolver
from ..objectives import ObjectiveBaseClass


class LimitedMemoryBfgs(ScipyBaseSolver):
    r"""Limited memory variant of BFGS (L-BFGS)

    See the documentation of scipy for the parameter list and description.

    Parameters
    ----------
    jac: None
        Is set automatically to objective gradient method. However, if no gradient function
        is available, e.g., for a custom distance function, then jac can be set to None.
    callback: callable
        Differently from the non-scipy solvers the signature is callback(xk) with xk the current
        set of variables, which are the  model parameters flattened to one 1D array.
    """

    def __init__(self, objective: ObjectiveBaseClass, **kwargs):
        super(LimitedMemoryBfgs, self).__init__(objective, method="L-BFGS-B", **kwargs)
