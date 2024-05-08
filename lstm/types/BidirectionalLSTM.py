from keras import Sequential
from keras.layers import *

from shared.AbstractNetwork import AbstractNetwork


class BidirectionalLSTM(AbstractNetwork):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        return Sequential([
            Bidirectional(LSTM(64, activation='tanh'), input_shape=(1997, 10)),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ])

    def getName(self):
        return "BidirectionalLSTM"
