
# Template and strategy design patterns.... Factory?
from abc import ABC, abstractmethod


# Template (Context)
class LVQClassifier(ABC):

    # But here I can have logic... because it's not a sklearn estimator?
    # Cannot change the value of the properties given in init...
    def __init__(self, objective, solver):
        self.objective = objective
        self.solver = solver

    # Can be overwritten - should do nothing in the base class. "hook"
    def pre_fit(self):
        self._variables = []
        self._cost_function_args = []

    def fit(self, data, labels):
        self.pre_fit()

        # Would this be a nice way to do this?
        objective = ObjectiveFactory.get_instance(self.objective)
        solver = SolverFactory.get_instance(self.solver)

        # Would it be possible to get the args from the objective function object, which can then be done in the solver
        # class. "objective_args = objective.get_args() returning some tuple"
        # or...
        # somehow bind solver to objective...

        self._variables = solver.solve(self._variables, objective.evaluate())


# Template (Context Implementation)
class GLVQClassifier(LVQClassifier):

    # How to vary objective function... its also strategy... cannot have logic in init function sklearn...
    def __init__(self, objective='default', solver='stochastic'):
        super(GLVQClassifier, self).__init__(objective, solver)

    def pre_fit(self):
        self._variables = [] # prototypes or in case of GMLVQ prototypes + omega
        self._objective_args = [] # This depends on the objective function used thus the context. also this depends on the objective function (My problem... I don't want to create a subclass for every LVQ algorithm with a differen costfunction...)


# ---------------------------------------------------------------------------------------------------------------------
# Strategy
class AbstractObjective(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError("You should implement this!")


# Strategy (Strategy Implementation)
class RelativeDistanceObjective(AbstractObjective):

    def __init__(self):
        super(RelativeDistanceObjective, self).__init__()

    # Override abstract method
    def evaluate(self):
        print('Calling RelativeDistanceObjective.evaluate()')


class ObjectiveFactory():

    @staticmethod
    def get_instance(self, objective_string):
        if objective_string == 'default':
            return RelativeDistanceObjective()


# ---------------------------------------------------------------------------------------------------------------------
# Strategy
class AbstractSolver(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def solve(self, *args, **kwargs):
        raise NotImplementedError("You should implement this!")


# Strategy (Stragey Implementation)
class StochasticSolver(AbstractSolver):

    def __init__(self):
        super(StochasticSolver, self).__init__()

    # Override abstract method
    def solve(self, *args, **kwargs):
        print('Calling StochasticSolver.solve()')


class SolverFactory():

    @staticmethod
    def get_instance(self, objective_string):
        if objective_string == 'stochastic':
            return StochasticSolver()