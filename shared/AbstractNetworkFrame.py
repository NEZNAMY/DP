import os
import threading
from abc import ABC, abstractmethod
from tkinter import Frame, Label

from options import Options
from shared import AbstractNetwork
from shared.frames.ModelConstructionFrame import ModelConstructionFrame
from shared.frames.ModelInfoFrame import ModelInfoFrame
from shared.frames.ModelTrainingFrame import ModelTrainingFrame


class AbstractNetworkFrame(ABC):

    def __init__(self, parentFrame: Frame, fullPath: str, network: AbstractNetwork, classCount: int):
        self.model = None
        self.modelImage = None
        self.fullPath = fullPath
        self.network = network
        self.frame = Frame(parentFrame)
        Options.instance.addFrame(self.frame)
        self.frame.columnconfigure(0, weight=2)
        self.frame.columnconfigure(1, weight=1)
        self.frame.columnconfigure(2, weight=1)
        self.frame.columnconfigure(3, weight=1)
        (Label(self.frame, text=network.getName() + " network", font=("Helvetica", 32))
         .grid(row=0, column=0, columnspan=99))
        self.modelFilePath = os.path.join(".networks", os.path.basename(fullPath), network.getName() + ".h5")

        self.modelInfoFrame = ModelInfoFrame(self.frame)
        self.modelInfoFrame.getFrame().grid(row=1, column=0, sticky="en", rowspan=99)

        self.constructionFrame = ModelConstructionFrame(self.frame, self.modelFilePath,
                                                        self.loadModel, network.createNetwork, classCount)
        self.constructionFrame.getFrame().grid(row=1, column=1, sticky="wn", columnspan=2, padx=15)

        self.trainingFrame = ModelTrainingFrame(self.frame, self.trainNetwork)
        self.trainingFrame.getFrame().grid(row=2, column=1, sticky="wn", padx=15, pady=15)

    def getFrame(self):
        return self.frame

    def loadModel(self, model):
        self.model = model
        self.trainingFrame.enableTrainButton()
        self.trainingFrame.hideWarn()
        self.modelInfoFrame.setModel(model, self.testAccuracy())

    def trainNetwork(self):
        def train():
            self.constructionFrame.disableButtons()
            self.train()
            self.trainingFrame.setTrainButtonText("Saving model...")
            self.model.save(self.modelFilePath)
            self.constructionFrame.checkSavedModel()
            self.trainingFrame.setTrainButtonText("Train")
            self.trainingFrame.enableTrainButton()
            self.constructionFrame.enableButtons()

        self.trainingFrame.disableTrainButton()
        threading.Thread(target=train).start()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def testAccuracy(self):
        pass
