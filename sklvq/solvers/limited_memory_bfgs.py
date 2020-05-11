from . import ScipyBaseSolver


class LimitedMemoryBfgs(ScipyBaseSolver):
    def __init__(self, params=None):
        super(LimitedMemoryBfgs, self).__init__(method="L-BFGS-B", params=params)
