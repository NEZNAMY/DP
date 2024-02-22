import keras

from keras.src.layers import *
from tensorflow.keras import *

from cnn.CNNFrame import *


class EfficientNetB4(AbstractCNN):

    def __init__(self, classCount: int):
        self.classCount = classCount

    def createNetwork(self):
        base_model = keras.applications.EfficientNetB4(weights='imagenet', include_top=False,
                                                       input_shape=(self.getImageSize(), self.getImageSize(), 3))

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
            Dense(self.classCount, activation='softmax')
        ], name="efficientnet_cnn_model")

    def getName(self):
        return "EfficientNetB4"

    def getImageSize(self):
        return 224
