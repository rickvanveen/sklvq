from . import ScipyBaseSolver


class BroydenFletcherGoldfarbShanno(ScipyBaseSolver):
    def __init__(self, params=None):
        super(BroydenFletcherGoldfarbShanno, self).__init__(method="BFGS", params=params)
