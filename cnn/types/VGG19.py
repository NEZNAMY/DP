import tensorflow
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

from cnn.CNNFrame import *


class VGG19(AbstractCNN):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        network = tensorflow.keras.applications.VGG19(weights='imagenet', include_top=False,
                                           input_shape=(self.getImageSize(), self.getImageSize(), 3))

        for layer in network.layers:
            layer.trainable = False

        return Sequential([
            network,
            Flatten(),
            Dense(4096, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ], name="vgg19_cnn_model")

    def getName(self):
        return "VGG19"

    def getImageSize(self):
        return 224
