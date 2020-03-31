from . import ScipyBaseSolver

# LimitedMemoryBFGS
class LbfgsSolver(ScipyBaseSolver):

    def __init__(self, params=None):
        super(LbfgsSolver, self).__init__(metqhod='L-BFGS - B', params=params)