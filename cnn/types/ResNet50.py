import tensorflow
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from cnn.CNNFrame import *


class ResNet50(AbstractCNN):

    def createNetwork(self, outputLayerSize: int, outputLayerActivation: str):
        base_model = tensorflow.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                                 input_shape=(self.getImageSize(), self.getImageSize(), 3))

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
            Dense(outputLayerSize, activation=outputLayerActivation)
        ], name="resnet50_cnn_model")

    def getName(self):
        return "ResNet50"

    def getImageSize(self):
        return 224
