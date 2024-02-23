from tkinter import Frame, Label

from PIL.ImageTk import PhotoImage

from options import Options
from shared import WrappedModel


class ModelInfoFrame:

    def __init__(self, parentFrame: Frame):
        self.model: WrappedModel = None
        self.frame = Frame(parentFrame)
        Options.instance.addFrame(self.frame)
        Label(self.frame, text="                                   "
                               "Model Info"
                               "                                   ",
              font=16).grid(row=0, column=0, columnspan=99)
        Label(self.frame, text="", font=16).grid(row=1, column=0, columnspan=99)

        # No model
        self.noModelWarn = Label(self.frame, text="No model is loaded", foreground="red")
        self.noModelWarn.grid(row=2, column=0, columnspan=99)

        # Model
        self.lossLabel = Label(self.frame, text="Loss function")
        self.lossValue = Label(self.frame, text="", foreground="green")
        self.optimizerLabel = Label(self.frame, text="Optimizer")
        self.optimizerValue = Label(self.frame, text="", foreground="green")
        self.trainAccuracyLabel = Label(self.frame, text="Train data accuracy")
        self.trainAccuracyValue = Label(self.frame, text="-", foreground="green")
        self.testAccuracyLabel = Label(self.frame, text="Test data accuracy")
        self.testAccuracyValue = Label(self.frame, text="-", foreground="green")
        self.modelInfoLabel = Label(self.frame)
        self.confusionMatrixLabel = Label(self.frame)

    def getFrame(self):
        return self.frame

    def setModel(self, model: WrappedModel, accuracy: list):
        if self.model is None:
            self.noModelWarn.grid_forget()
            self.lossLabel.grid(row=2, column=1)
            self.lossValue.grid(row=2, column=2)
            self.optimizerLabel.grid(row=3, column=1)
            self.optimizerValue.grid(row=3, column=2)
            self.trainAccuracyLabel.grid(row=4, column=1)
            self.trainAccuracyValue.grid(row=4, column=2)
            self.testAccuracyLabel.grid(row=5, column=1)
            self.testAccuracyValue.grid(row=5, column=2)
            self.modelInfoLabel.grid(row=2, column=0, rowspan=99)
            self.confusionMatrixLabel.grid(row=6, column=1, columnspan=99, rowspan=99, sticky="n")

        self.model = model
        self.updateModelInfo(accuracy)

    def updateModelInfo(self, accuracy: list):
        self.lossValue.config(text=self.model.getLossFunction())
        self.optimizerValue.config(text=self.model.getOptimizer())
        self.setTrainAccuracy(accuracy[0])
        self.setTestAccuracy(accuracy[1])
        self.modelInfoLabel.config(image=self.model.getModelImage())

    def setConfusionMatrix(self, img: PhotoImage):
        self.confusionMatrixLabel.img = img  # Keep a reference to avoid garbage collection
        self.confusionMatrixLabel.config(image=img)

    def setTrainAccuracy(self, accuracy: int):
        if accuracy == 0:
            self.trainAccuracyValue.config(text="-")
        else:
            self.trainAccuracyValue.config(text=format(accuracy*100, ".2f") + "%")

    def setTestAccuracy(self, accuracy: int):
        if accuracy == 0:
            self.testAccuracyValue.config(text="-")
        else:
            self.testAccuracyValue.config(text=format(accuracy*100, ".2f") + "%")
