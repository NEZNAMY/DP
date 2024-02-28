import tensorflow
from keras.layers import *
from tensorflow.keras import *

from cnn.CNNFrame import *


class InceptionV3(AbstractCNN):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        network = tensorflow.keras.applications.InceptionV3(include_top=False, weights="imagenet",
                                                 input_shape=(self.getImageSize(), self.getImageSize(), 3))
        for layer in network.layers:
            layer.trainable = False

        return Sequential([
            network,
            Dropout(0.5),
            GlobalAveragePooling2D(),
            Flatten(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ], name="inception_cnn_model")

    def getName(self):
        return "InceptionV3"

    def getImageSize(self):
        return 299
