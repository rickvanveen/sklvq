from . import ScipyBaseSolver


class LbfgsSolver(ScipyBaseSolver):

    def __init__(self):
        super(LbfgsSolver, self).__init__(method='L-BFGS-B')