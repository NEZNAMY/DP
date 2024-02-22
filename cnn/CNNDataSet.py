import glob
import os
from tkinter import *

from cnn.CNNFrame import CNNFrame
from cnn.CNNInfoFrame import CNNInfoFrame
from cnn.types.DenseNet121 import DenseNet121
from cnn.types.EfficientNetB4 import EfficientNetB4
from cnn.types.InceptionV3 import InceptionV3
from cnn.types.MobileNetV2 import MobileNetV2
from cnn.types.ResNet50 import ResNet50
from cnn.types.VGG16 import VGG16
from cnn.types.VGG19 import VGG19
from cnn.types.Xception import Xception
from shared.AbstractDataSet import AbstractDataSet


class CNNDataSet(AbstractDataSet):

    def __init__(self, tk: Tk, fullPath: str, displayName: str):
        super().__init__(tk, fullPath, displayName)

    def createInfoFrame(self):
        images = {}
        for category in self.categories:
            images[category] = glob.glob(os.path.join(self.fullPath, category, "*"))
        self.info = CNNInfoFrame(self.frame, self.displayName, images)
        return self.info.getFrame()

    def loadCategories(self):
        return [f for f in os.listdir(self.fullPath) if
                os.path.isdir(os.path.join(self.fullPath, f)) and f not in ["train", "test"]]

    def loadNetworks(self):
        for cnn in [DenseNet121(), EfficientNetB4(), InceptionV3(),
                    MobileNetV2(), ResNet50(), VGG16(), VGG19(), Xception()]:
            cnf = CNNFrame(self.trainingFrame, self.fullPath, self.classIndexMapping, cnn)
            self.networks.addNetwork(cnn.getName(), cnf.getFrame())
