from abc import ABC, abstractmethod


class ObjectiveBaseClass(ABC):

    @abstractmethod
    def evaluate(self, data, labels, prototypes, prototype_labels):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, data, labels, prototypes, prototype_labels):
        raise NotImplementedError("You should implement this!")
