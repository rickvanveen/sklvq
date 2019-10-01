from abc import ABC, abstractmethod


class ActivationBaseClass(ABC):

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, x):
        raise NotImplementedError("You should implement this!")
