from . import ScipyBaseSolver
from ..objectives import ObjectiveBaseClass


class BroydenFletcherGoldfarbShanno(ScipyBaseSolver):
    r"""Broyden Fletcher Goldfarb Shanno (BFGS)

    See the documentation of scipy for a complete parameter list and description.

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
        super(BroydenFletcherGoldfarbShanno, self).__init__(
            objective, method="BFGS", **kwargs
        )
