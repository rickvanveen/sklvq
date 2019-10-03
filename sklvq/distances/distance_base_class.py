from abc import ABC, abstractmethod


class DistanceBaseClass(ABC):

    @abstractmethod
    def __call__(self, data, model):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, data, model, i_prototype):
        raise NotImplementedError("You should implement this!")