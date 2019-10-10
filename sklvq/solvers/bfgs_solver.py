from . import ScipyBaseSolver


class BfgsSolver(ScipyBaseSolver):

    def __init__(self):
        super(BfgsSolver, self).__init__(method='BFGS')
