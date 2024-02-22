from keras import Sequential
from keras.src.layers import Dense, Dropout

from shared.AbstractNetwork import AbstractNetwork


class MLPNetwork(AbstractNetwork):

    def __init__(self, featureCount: int, structure: dict):
        self.featureCount = featureCount
        self.structure = structure

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        model = Sequential()
        layers = self.structure["Layers"]
        for index, layer in enumerate(layers):
            first = index == 0
            layerType = layer["LayerType"]
            parameter = layer["Parameter"]
            if layerType == "Dense":
                activation = layer["ActivationFunction"]
                model.add(self.dense(int(parameter), activation, first))
            elif layerType == "Dropout":
                model.add(self.dropout(float(parameter), first))
        model.add(Dense(outputLayerSize, activation=outputLayerActivation))
        return model

    def dense(self, count, activation, first):
        if first:
            return Dense(count, activation=activation, input_dim=self.featureCount)
        else:
            return Dense(count, activation=activation)

    def dropout(self, ratio, first):
        if first:
            return Dropout(ratio, input_dim=self.featureCount)
        else:
            return Dropout(ratio)

    def getName(self):
        return self.structure["Name"]
