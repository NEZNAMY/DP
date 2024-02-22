from abc import ABC, abstractmethod


class AbstractNetwork(ABC):

    @abstractmethod
    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        pass

    @abstractmethod
    def getName(self):
        pass
