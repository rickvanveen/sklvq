from . import ScipyBaseSolver


class BroydenFletcherGoldfarbShanno(ScipyBaseSolver):
    """ BroydenFletcherGoldfarbShanno

    """
    def __init__(self, objective, **kwargs):
        super(BroydenFletcherGoldfarbShanno, self).__init__(
            objective, method="BFGS", **kwargs
        )
