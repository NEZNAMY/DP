from keras import Sequential
from keras.layers import *

from shared.AbstractNetwork import AbstractNetwork


class CNNLSTM(AbstractNetwork):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        return Sequential([
            Conv1D(filters=64, kernel_size=3, activation='tanh', input_shape=(1997, 10)),
            MaxPooling1D(pool_size=2),
            LSTM(64),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ])

    def getName(self):
        return "CNNLSTM"
