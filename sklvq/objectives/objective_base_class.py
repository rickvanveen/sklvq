from abc import ABC, abstractmethod


class ObjectiveBaseClass(ABC):

    @abstractmethod
    def cost(self, data, labels, model):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, data, labels, model):
        raise NotImplementedError("You should implement this!")
