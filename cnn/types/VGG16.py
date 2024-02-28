import tensorflow

from keras.layers import *
from tensorflow.keras import *

from cnn.CNNFrame import *


class VGG16(AbstractCNN):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        network = tensorflow.keras.applications.VGG16(weights='imagenet', include_top=False,
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
        ], name="vgg16_cnn_model")

    def getName(self):
        return "VGG16"

    def getImageSize(self):
        return 224
