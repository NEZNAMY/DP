import tensorflow

from keras.layers import *
from tensorflow.keras import *

from cnn.CNNFrame import *


class EfficientNetB4(AbstractCNN):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        base_model = tensorflow.keras.applications.EfficientNetB4(
            include_top=False, weights='imagenet', input_shape=(self.getImageSize(), self.getImageSize(), 3))

        for layer in base_model.layers:
            layer.trainable = False

        return Sequential([
            base_model,
            Flatten(),
            Dense(4096, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ], name="efficientnet_cnn_model")

    def getName(self):
        return "EfficientNetB4"

    def getImageSize(self):
        return 224
