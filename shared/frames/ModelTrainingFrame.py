import threading
from tkinter import Label, Entry, Button, Frame
from options import Options
from shared import AbstractNetworkFrame


class ModelTrainingFrame:

    def __init__(self, parentFrame: Frame, networkFrame: AbstractNetworkFrame):
        self.networkFrame = networkFrame
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

        self.trainButton = Button(self.frame, text="Train", command=self.trainNetwork)
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

    def trainNetwork(self):
        def train():
            self.networkFrame.constructionFrame.disableButtons()
            self.networkFrame.train()
            self.setSavingModel()
            self.networkFrame.model.getModel().save(self.networkFrame.modelFilePath)
            self.networkFrame.constructionFrame.modelLoading.checkSavedModel()
            self.setTrainButtonText("Train")
            self.enableTrainButton()
            self.networkFrame.constructionFrame.enableButtons()

        self.disableTrainButton()
        threading.Thread(target=train).start()

    def setStartingPhaseText(self):
        self.setTrainButtonText("Phase 1/6: Starting...")

    def setTrainingPhaseText(self, progress: int):
        self.setTrainButtonText("Phase 2/6: Training... (" + str(progress) + "%)")

    def setTestingTrainData(self):
        self.setTrainButtonText("Phase 3/6: Testing accuracy on training data...")

    def setTestingTestData(self):
        self.setTrainButtonText("Phase 4/6: Testing accuracy on testing data...")

    def setCreatingConfusionMatrix(self):
        self.setTrainButtonText("Phase 5/6: Creating confusion matrix...")

    def setSavingModel(self):
        self.setTrainButtonText("Phase 6/6: Saving model...")
