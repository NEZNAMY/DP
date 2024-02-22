from abc import ABC, abstractmethod


class AbstractNetwork(ABC):

    @abstractmethod
    def createNetwork(self):
        pass

    @abstractmethod
    def getName(self):
        pass
