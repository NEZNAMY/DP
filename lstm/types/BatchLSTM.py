from keras import Sequential
from keras.layers import *

from shared.AbstractNetwork import AbstractNetwork


class BatchLSTM(AbstractNetwork):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        return Sequential([
            LSTM(64, activation='relu', input_shape=(1997, 5)),
            BatchNormalization(),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ])

    def getName(self):
        return "BatchLSTM"
