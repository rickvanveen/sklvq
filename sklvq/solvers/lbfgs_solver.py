from . import SolverBaseClass

import scipy as sp


class LbfgsSolver(SolverBaseClass):

    # TODO: Need to accept parameters of the scipy function fmin_l_bfgs_b
    # def __init__(self):
    #     pass

    def solve(self, data, labels, objective, model):
        result = sp.optimize.fmin_l_bfgs_b(objective, model.get_variables(),
                                           fprime=objective.gradient, args=(model, data, labels))
        model.set_variables(result[0])
        return model
