from keras import Sequential
from keras.layers import *

from shared.AbstractNetwork import AbstractNetwork


class MyLSTM(AbstractNetwork):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        return Sequential([
            LSTM(50, activation='relu', input_shape=(1, 1)),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ])

    def getName(self):
        return "MyLSTM"
