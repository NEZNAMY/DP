import os
from abc import ABC, abstractmethod
from tkinter import Frame, Label, Button

import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from options import Options
from shared import AbstractNetwork
from shared.WrappedModel import WrappedModel
from shared.frames.construction.ModelConstructionFrame import ModelConstructionFrame
from shared.frames.ModelInfoFrame import ModelInfoFrame
from shared.frames.ModelTrainingFrame import ModelTrainingFrame


class AbstractNetworkFrame(ABC):

    def __init__(self, parentFrame: Frame, fullPath: str, network: AbstractNetwork, classCount: int):
        self.confusionMatrixLabel = None
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

        self.constructionFrame = ModelConstructionFrame(self, self.modelFilePath,
                                                        self.loadModel, network.createNetwork, classCount)
        self.constructionFrame.getFrame().grid(row=1, column=1, sticky="wn", columnspan=2, padx=15)

        self.trainingFrame = ModelTrainingFrame(self.frame, self)
        self.trainingFrame.getFrame().grid(row=2, column=1, sticky="wn", padx=15, pady=15)

    def getFrame(self):
        return self.frame

    def loadModel(self, model: WrappedModel, button: Button):
        self.model = model
        self.trainingFrame.enableTrainButton()
        self.trainingFrame.hideWarn()
        button.config(text="Phase 2/4: Splitting data to train/test...")
        self.checkSplitData()
        button.config(text="Phase 3/4: Testing model accuracy...")
        self.modelInfoFrame.setModel(model, self.testAccuracy())
        button.config(text="Phase 4/4: Creating confusion matrix...")
        self.updateConfusionMatrix()

    def updateConfusionMatrix(self):
        disp = self.createConfusionMatrix()
        if disp is None:
            return

        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        disp.plot(cmap='Blues', ax=ax)
        ax.get_figure().delaxes(ax.figure.axes[-1])

        plt.xticks(rotation=80)
        plt.xlabel('Predicted', color='red')
        plt.ylabel('True', color='blue')
        plt.title('Confusion Matrix')

        fig.tight_layout()
        img = self.fig2img(fig)
        plt.close(fig)
        if img is None:
            return
        self.modelInfoFrame.setConfusionMatrix(img)

    def fig2img(self, fig: Figure):
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.get_renderer().buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = Image.fromarray(img_data, 'RGBA')
        return ImageTk.PhotoImage(img)

    @abstractmethod
    def createConfusionMatrix(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def testAccuracy(self):
        pass

    def checkSplitData(self):
        pass
