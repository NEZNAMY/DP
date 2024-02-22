import os
from tkinter import Tk, Button

import pandas as pd

from config import Config
from mlp.MLPConstructionMenu import MLPConstructionMenu
from mlp.MLPFrame import MLPFrame
from mlp.MLPInfoFrame import MLPInfoFrame
from mlp.MLPNetwork import MLPNetwork
from shared.AbstractDataSet import AbstractDataSet


class MLPDataSet(AbstractDataSet):

    def __init__(self, tk: Tk, fullPath: str, displayName: str):
        super().__init__(tk, fullPath, displayName)

    def createInfoFrame(self):
        self.info = MLPInfoFrame(self.frame, self.displayName, self.features, self.classes)
        return self.info.getFrame()

    def loadDataSet(self):
        self.features = pd.read_csv(os.path.join(self.fullPath, 'features.csv'))
        self.featureCount = len(self.features.columns.tolist())
        self.target = pd.read_csv(os.path.join(self.fullPath, 'targets.csv'))
        self.classCount = self.target['target'].nunique()
        self.classes = self.target['target'].value_counts().to_dict()

    def loadNetworks(self):
        for structure in Config.instance.getMLPStructures():
            self.addNetwork(structure)

        Button(self.networks.frame, text="+ New network structure", foreground="green", width=25,
               command=lambda: MLPConstructionMenu(Config.instance.getMLPStructureNames(), self.addNetwork)).grid(
            row=0, column=99, sticky="nsew")

    def addNetwork(self, structure: dict):
        net = MLPNetwork(self.featureCount, structure)
        mlp = MLPFrame(self.trainingFrame, self, net)
        self.networks.addNetwork(net.getName(), mlp.getFrame())
