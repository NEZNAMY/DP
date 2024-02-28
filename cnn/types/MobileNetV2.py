import tensorflow
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

from cnn.CNNFrame import *


class MobileNetV2(AbstractCNN):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        base_model = tensorflow.keras.applications.MobileNetV2(weights='imagenet', include_top=False,
                                                    input_shape=(self.getImageSize(), self.getImageSize(), 3))

        for layer in base_model.layers:
            layer.trainable = False

        return Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(outputLayerSize, activation=outputLayerActivation)
        ], name="mobilenetv2_cnn_model")

    def getName(self):
        return "MobileNetV2"

    def getImageSize(self):
        return 224
