from abc import ABC, abstractmethod

import scipy as sp


class AbstractSolver(ABC):

    def __init__(self, objective):
        self.objective = objective

    @abstractmethod
    def solve(self, *args, **kwargs):
        raise NotImplementedError("You should implement this!")


#  TODO: How should this work now... because it's an explicit solver and specific for the algorithm.
class StochasticSolver(AbstractSolver):

    def __init__(self):
        super(StochasticSolver, self).__init__()

    def solve(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet")


class LBFGSBSolver(AbstractSolver):

    def __init__(self, objective=None):
        super(LBFGSBSolver, self).__init__(objective)

    def solve(self, variables, objective_args):
        result = sp.optimize.fmin_l_bfgs_b(self.objective, variables, args=objective_args)
        return result[0]


class SolverFactory:

    @staticmethod
    def create(solver_type):
        if solver_type == 'stochastic':
            return StochasticSolver()
        elif solver_type == 'l-bfgs-b':
            return LBFGSBSolver()
        else:
            print("Solver type does not exist")
