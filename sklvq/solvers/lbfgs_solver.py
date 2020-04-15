from . import ScipyBaseSolver


# LimitedMemoryBFGS
class LbfgsSolver(ScipyBaseSolver):
    def __init__(self, params=None):
        super(LbfgsSolver, self).__init__(method="L-BFGS-B", params=params)
