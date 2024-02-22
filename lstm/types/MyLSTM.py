from keras import Sequential
from keras.src.layers import *

from shared.AbstractNetwork import AbstractNetwork


class MyLSTM(AbstractNetwork):

    def createNetwork(self):
        return Sequential([
            LSTM(50, activation='relu', input_shape=(1, 1)),
            Dense(1)
        ])

    def getName(self):
        return "MyLSTM"
