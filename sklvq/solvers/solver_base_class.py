from abc import ABC, abstractmethod


class SolverBaseClass(ABC):

    @abstractmethod
    def solve(self, data, labels, objective, model):
        raise NotImplementedError("You should implement this!")
