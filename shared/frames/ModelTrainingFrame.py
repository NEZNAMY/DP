from tkinter import Label, Entry, Button, Frame
from typing import Callable
from options import Options


class ModelTrainingFrame:

    def __init__(self, parentFrame: Frame, trainCommand: Callable):
        self.frame = Frame(parentFrame)
        Options.instance.addFrame(self.frame)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        Label(self.frame, text="Training menu", font=16).grid(row=0, column=0, columnspan=99)
        Label(self.frame, text="", font=16).grid(row=1, column=0, columnspan=99)
        Label(self.frame, text="Epochs").grid(row=2, column=0)

        self.epochEntry = Entry(self.frame, width=10)
        self.epochEntry.insert(0, "5")
        self.epochEntry.grid(row=2, column=1)
        Label(self.frame, text="Batch size").grid(row=3, column=0)

        self.batchEntry = Entry(self.frame, width=10)
        self.batchEntry.insert(0, "32")
        self.batchEntry.grid(row=3, column=1)

        self.trainButton = Button(self.frame, text="Train", command=trainCommand)
        self.trainButton.grid(row=4, column=0, columnspan=2)
        self.trainButton.config(state="disabled")

        self.warnLabel = Label(self.frame, text="Training is not available: No model is loaded", foreground="red")
        self.warnLabel.grid(row=5, column=0, columnspan=2)

    def getFrame(self):
        return self.frame

    def enableTrainButton(self):
        self.trainButton.config(state="normal")

    def disableTrainButton(self):
        self.trainButton.config(state="disabled")

    def setTrainButtonText(self, text: str):
        self.trainButton.config(text=text)

    def getBatchSize(self):
        return int(self.batchEntry.get())

    def getEpochCount(self):
        return int(self.epochEntry.get())

    def hideWarn(self):
        self.warnLabel.config(text="")
