from keras import Sequential
from keras.layers import *

from shared.AbstractNetwork import AbstractNetwork


class CNNLSTM(AbstractNetwork):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        return Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(None, None)),
            MaxPooling1D(pool_size=2),
            LSTM(64),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ])

    def getName(self):
        return "CNNLSTM"
