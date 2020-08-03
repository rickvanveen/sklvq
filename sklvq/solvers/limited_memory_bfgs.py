from . import ScipyBaseSolver


class LimitedMemoryBfgs(ScipyBaseSolver):
    """ LimitedMemoryBfgs

    """
    def __init__(self, objective, **kwargs):
        super(LimitedMemoryBfgs, self).__init__(
            objective, method="L-BFGS-B", **kwargs
        )
