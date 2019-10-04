from abc import ABC, abstractmethod


class ObjectiveBaseClass(ABC):

    @abstractmethod
    def cost(self, variables, data, labels, model):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, variables, data, labels, model):
        raise NotImplementedError("You should implement this!")
