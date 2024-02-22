import glob
import os
from tkinter import *

from cnn.types import AbstractCNN
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
        self.classIndexMapping = {}
        self.images = {}
        self.categories = []
        self.class_weights = {}
        super().__init__(tk, fullPath, displayName)

    def createInfoFrame(self):
        self.info = CNNInfoFrame(self.frame, self.displayName, self.images)
        return self.info.getFrame()

    def loadDataSet(self):
        self.categories = [f for f in os.listdir(self.fullPath) if
                           os.path.isdir(os.path.join(self.fullPath, f)) and f not in ["train", "test"]]
        for category in self.categories:
            self.images[category] = glob.glob(os.path.join(self.fullPath, category, "*"))

        total_samples = sum(len(lst) for lst in self.images.values())
        self.classIndexMapping = {i: class_name for i, class_name in enumerate(self.categories)}
        i = 0
        for value in self.images.values():
            self.class_weights[i] = total_samples / len(value),
            i += 1

    def loadNetworks(self):
        self.addNetwork(DenseNet121())
        self.addNetwork(EfficientNetB4())
        self.addNetwork(InceptionV3())
        self.addNetwork(MobileNetV2())
        self.addNetwork(ResNet50())
        self.addNetwork(VGG16())
        self.addNetwork(VGG19())
        self.addNetwork(Xception())

    def addNetwork(self, cnn: AbstractCNN):
        cnf = CNNFrame(self.trainingFrame, self.fullPath, self.class_weights, self.classIndexMapping, cnn)
        self.networks.addNetwork(cnn.getName(), cnf.getFrame())
