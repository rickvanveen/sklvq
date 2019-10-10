from . import SolverBaseClass

import scipy as sp


class ScipyBaseSolver(SolverBaseClass):

    def __init__(self, method='L-BFGS-B'):
        self.method = method

    def solve(self, data, labels, objective, model):
        result = sp.optimize.minimize(objective, model.get_variables(), method=self.method,
                                      jac=objective.gradient, args=(model, data, labels))
        model.set_variables(result.x)
        return model
