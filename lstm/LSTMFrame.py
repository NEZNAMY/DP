from tkinter import Frame

import numpy as np
import matplotlib.pyplot as plt
from lstm import LSTMDataSet
from shared import AbstractNetwork
from shared.AbstractNetworkFrame import AbstractNetworkFrame


class LSTMFrame(AbstractNetworkFrame):

    def __init__(self, parentFrame: Frame, dataSet: LSTMDataSet, network: AbstractNetwork):
        super().__init__(parentFrame, dataSet.fullPath, network, False)
        self.dataSet = dataSet

    def train(self):
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(
            self.dataSet.dataX,
            self.dataSet.dataY,
            epochs=self.trainingFrame.getEpochCount(),
            batch_size=self.trainingFrame.getBatchSize(),
        )

        # Generate predictions
        x_test = np.linspace(10, 20, 50)
        x_test = x_test.reshape((50, 1, 1))
        y_pred = self.model.predict(x_test)

        # Plot the results
        plt.plot(self.dataSet.dataX[:, 0, 0], self.dataSet.dataY, label='Original Data')
        plt.plot(x_test[:, 0, 0], y_pred, label='Predictions', linestyle='dashed')
        plt.legend()
        plt.show()

    def testAccuracy(self):
        return [0, 0]  # TODO

    def createConfusionMatrix(self):
        return None
