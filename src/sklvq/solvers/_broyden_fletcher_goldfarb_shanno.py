from sklvq.objectives._base import ObjectiveBaseClass
from sklvq.solvers._base import ScipyBaseSolver


class BroydenFletcherGoldfarbShanno(ScipyBaseSolver):
    r"""Broyden Fletcher Goldfarb Shanno (BFGS)

    See the documentation of scipy for a complete parameter list and description.

    Parameters
    ----------
    jac: None
        Is set automatically to objective gradient method. However, if no gradient function
        is available, e.g., for a custom distance function, then jac can be set to None.
    callback: callable
        Called with a detached state dictionary before training and at each accepted iterate.
        It contains model_params (named parameter arrays), nit ("Initial", then 1, 2, ...),
        fun (objective cost), and max_runs (the maxiter limit). Return True to stop.
    """

    def __init__(self, objective: ObjectiveBaseClass, **kwargs):
        super().__init__(objective, method="BFGS", **kwargs)
