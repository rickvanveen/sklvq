from . import ScipyBaseSolver


class LimitedMemoryBfgs(ScipyBaseSolver):
    def __init__(self, objective, params=None):
        super(LimitedMemoryBfgs, self).__init__(
            objective, method="L-BFGS-B", params=params
        )
