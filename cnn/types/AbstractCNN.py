from abc import abstractmethod

from shared.AbstractNetwork import AbstractNetwork


class AbstractCNN(AbstractNetwork):

    @abstractmethod
    def getImageSize(self):
        pass
