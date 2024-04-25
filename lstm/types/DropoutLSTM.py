from keras import Sequential
from keras.layers import *

from shared.AbstractNetwork import AbstractNetwork


class DropoutLSTM(AbstractNetwork):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        return Sequential([
            LSTM(64, activation='relu', input_shape=(None, None)),
            Dropout(0.2),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ])

    def getName(self):
        return "DropoutLSTM"