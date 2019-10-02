from abc import ABC, abstractmethod


class DiscriminativeBaseClass(ABC):

    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, *args):
        raise NotImplementedError("You should implement this!")