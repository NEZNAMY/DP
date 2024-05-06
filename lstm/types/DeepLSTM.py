from keras import Sequential
from keras.layers import *

from shared.AbstractNetwork import AbstractNetwork


class DeepLSTM(AbstractNetwork):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        return Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=(1997, 5)),
            LSTM(64, return_sequences=True),
            LSTM(64),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ])

    def getName(self):
        return "DeepLSTM"
