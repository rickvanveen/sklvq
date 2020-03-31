from . import ScipyBaseSolver

# TODO Rename Bfgs...
# BroydenFletcherGoldfarbShanno
# LimitedMemoryBFGS
class BfgsSolver(ScipyBaseSolver):

    def __init__(self, params=None):
        super(BfgsSolver, self).__init__(method='BFGS', params=params)
