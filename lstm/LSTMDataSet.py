import os
from tkinter import Tk

from lstm.LSTMFrame import LSTMFrame
from lstm.LSTMInfoFrame import LSTMInfoFrame
from lstm.types.BatchLSTM import BatchLSTM
from lstm.types.BidirectionalLSTM import BidirectionalLSTM
from lstm.types.CNNLSTM import CNNLSTM
from lstm.types.DeepLSTM import DeepLSTM
from lstm.types.DropoutLSTM import DropoutLSTM
from lstm.types.SingleLSTM import SingleLSTM
from lstm.types.StackedLSTM import StackedLSTM
from shared import AbstractNetwork
from shared.AbstractDataSet import AbstractDataSet


class LSTMDataSet(AbstractDataSet):

    def __init__(self, tk: Tk, fullPath: str, displayName: str):
        self.classes = {}
        self.fullPath = fullPath
        for name in self.loadCategories():
            directory = os.path.join(fullPath, name)
            self.classes[name] = len(
                    [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
        super().__init__(tk, fullPath, displayName)

    def createInfoFrame(self):
        self.info = LSTMInfoFrame(self.frame, self.fullPath, self.displayName, self.classes)
        return self.info.getFrame()

    def loadCategories(self):
        directory_list = []
        for directory in os.listdir(self.fullPath):
            directory_path = os.path.join(self.fullPath, directory)
            if os.path.isdir(directory_path):
                directory_list.append(directory)
        return directory_list

    def loadNetworks(self):
        self.addNetwork(SingleLSTM())
        self.addNetwork(StackedLSTM())
        self.addNetwork(BidirectionalLSTM())
        self.addNetwork(CNNLSTM())
        self.addNetwork(DeepLSTM())
        self.addNetwork(DropoutLSTM())
        self.addNetwork(BatchLSTM())

    def addNetwork(self, lstm: AbstractNetwork):
        lnf = LSTMFrame(self.trainingFrame, self, lstm)
        self.networks.addNetwork(lstm.getName(), lnf.getFrame())
