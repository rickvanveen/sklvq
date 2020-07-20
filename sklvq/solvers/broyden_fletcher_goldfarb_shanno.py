from . import ScipyBaseSolver


class BroydenFletcherGoldfarbShanno(ScipyBaseSolver):
    def __init__(self, objective, params=None):
        super(BroydenFletcherGoldfarbShanno, self).__init__(
            objective, method="BFGS", params=params
        )
