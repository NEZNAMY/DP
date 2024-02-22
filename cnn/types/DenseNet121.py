import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

from cnn.types.AbstractCNN import AbstractCNN


class DenseNet121(AbstractCNN):

    def __init__(self, classCount: int):
        self.classCount = classCount

    def createNetwork(self):
        base_model = keras.applications.DenseNet121(
            include_top=False, weights='imagenet', input_shape=(self.getImageSize(), self.getImageSize(), 3))

        for layer in base_model.layers:
            layer.trainable = False

        return Sequential([
            base_model,
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
            Dense(self.classCount, activation='softmax')
        ], name="densenet121_cnn_model")

    def getName(self):
        return "DenseNet121"

    def getImageSize(self):
        return 224
