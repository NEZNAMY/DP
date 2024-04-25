from keras import Sequential
from keras.layers import *

from shared.AbstractNetwork import AbstractNetwork


class SingleLSTM(AbstractNetwork):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        return Sequential([
            LSTM(64, activation='relu', input_shape=(None, None)),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ])

    def getName(self):
        return "SingleLSTM"
