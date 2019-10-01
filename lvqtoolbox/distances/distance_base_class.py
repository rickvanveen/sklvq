from abc import ABC, abstractmethod


class DistanceBaseClass(ABC):

    @abstractmethod
    def __call__(self, data, prototypes):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, data, prototype):
        raise NotImplementedError("You should implement this!")