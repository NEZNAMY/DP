import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

from cnn.CNNFrame import *


class Xception(AbstractCNN):

    def __init__(self, classCount: int):
        self.classCount = classCount

    def createNetwork(self):
        base_model = keras.applications.Xception(weights='imagenet', include_top=False,
                                                 input_shape=(self.getImageSize(), self.getImageSize(), 3))

        for layer in base_model.layers:
            layer.trainable = False

        return Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.classCount, activation='softmax')
        ], name="xception_cnn_model")

    def getName(self):
        return "Xception"

    def getImageSize(self):
        return 299
