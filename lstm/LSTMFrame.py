from tkinter import Frame

import tensorflow as tf

from lstm import LSTMDataSet
from shared import AbstractNetwork
from shared.AbstractNetworkFrame import AbstractNetworkFrame


class LSTMFrame(AbstractNetworkFrame):

    def __init__(self, parentFrame: Frame, dataSet: LSTMDataSet, network: AbstractNetwork):
        super().__init__(parentFrame, dataSet.fullPath, network, False)
        self.dataSet = dataSet

    def train(self):
        self.model.getModel().fit(
            self.dataSet.dataX,
            self.dataSet.dataY,
            epochs=self.trainingFrame.getEpochCount(),
            batch_size=self.trainingFrame.getBatchSize(),
        )

    def testAccuracy(self):
        return [0, 0]  # TODO

    def createConfusionMatrix(self):
        return None


class CustomLSTMCallback(tf.keras.callbacks.Callback):

    def __init__(self, anf: AbstractNetworkFrame, totalEpochs: int):
        super(CustomLSTMCallback, self).__init__()
        self.anf = anf
        self.totalEpochs = totalEpochs

    def on_epoch_end(self, epoch, logs=None):
        percent = int((epoch + 1) / self.totalEpochs * 100)
        self.anf.trainingFrame.setTrainingPhaseText(percent)
