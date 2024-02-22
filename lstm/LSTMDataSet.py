import os
from tkinter import Tk

import numpy as np

from lstm.LSTMFrame import LSTMFrame
from lstm.LSTMInfoFrame import LSTMInfoFrame
from lstm.types.MyLSTM import MyLSTM
from shared import AbstractNetwork
from shared.AbstractDataSet import AbstractDataSet


class LSTMDataSet(AbstractDataSet):

    def loadCategories(self):
        return []

    def __init__(self, tk: Tk, fullPath: str, displayName: str):
        super().__init__(tk, fullPath, displayName)

    def createInfoFrame(self):
        self.info = LSTMInfoFrame(self.frame, self.displayName)
        return self.info.getFrame()

    def loadDataSet(self):
        loaded_data = np.load(os.path.join(self.fullPath, 'data.npz'))
        self.dataX, self.dataY = loaded_data['x'], loaded_data['y']

    def loadNetworks(self):
        self.addNetwork(MyLSTM())

    def addNetwork(self, lstm: AbstractNetwork):
        lnf = LSTMFrame(self.trainingFrame, self, lstm)
        self.networks.addNetwork(lstm.getName(), lnf.getFrame())
