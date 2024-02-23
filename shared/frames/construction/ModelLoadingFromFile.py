import os
import threading
import time
from tkinter import Frame, Label, Button

from tensorflow.keras.models import load_model

from options import Options
from shared.frames.construction import ModelConstructionFrame


class ModelLoadingFromFile:

    def __init__(self, parent: ModelConstructionFrame, modelFilePath: str):
        self.modelConstruction = parent
        self.modelFilePath = modelFilePath

        self.frame = Frame(parent.getFrame())
        Options.instance.addFrame(self.frame)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)

        Label(self.frame, text="Load model from file").grid(row=0, column=0)
        Label(self.frame, text="").grid(row=1, column=0, columnspan=2)
        self.loadButton = Button(self.frame, text="Load from file", command=self.loadFromFile)
        self.loadButton.grid(row=4, column=0)
        self.loadButton.config(state="disabled")
        self.foundLabel = Label(self.frame, text="No saved model was found", foreground="red")
        self.checkSavedModel()
        self.foundLabel.grid(row=3, column=0)

    def getFrame(self):
        return self.frame

    def loadFromFile(self):
        def load():
            self.loadButton.config(state="disabled", text="Phase 1/3: Loading model to memory...")
            model = load_model(self.modelFilePath)
            self.modelConstruction.parent.loadModel(model, self.loadButton)
            self.loadButton.config(state="normal", text="Load from file")
            print("Loaded model from file in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        threading.Thread(target=load).start()

    def checkSavedModel(self):
        if os.path.exists(self.modelFilePath):
            self.loadButton.config(state="normal")
            size = format(os.path.getsize(self.modelFilePath) / 1024 / 1024, '.2f') + " MB"
            self.foundLabel.config(text="Saved model available (" + size + ")", foreground="green")
