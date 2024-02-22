import os
import threading
from tkinter import Label, Button, Frame
from tkinter.ttk import Combobox
from typing import Callable

import tensorflow
import tensorflow as tf

from options import Options


class ModelConstructionFrame:

    def __init__(self, parentFrame: Frame, modelFilePath: str, loadModel: Callable, createModel: Callable,
                 allowBinary: bool):
        self.modelFilePath = modelFilePath
        self.loadModel = loadModel
        self.createModel = createModel
        self.constructionFrame = Frame(parentFrame)
        Options.instance.addFrame(self.constructionFrame)
        Label(self.constructionFrame, text="Construction menu", font=16).grid(row=0, column=0, columnspan=99)
        self.constructionFrame.columnconfigure(0, weight=1)
        self.constructionFrame.columnconfigure(1, weight=1)

        # Create
        self.createFrame = Frame(self.constructionFrame)
        Options.instance.addFrame(self.createFrame)
        self.createFrame.grid(row=1, column=0)
        self.createFrame.columnconfigure(0, weight=1)
        self.createFrame.columnconfigure(1, weight=1)
        Label(self.createFrame, text="Create new model").grid(row=1, column=0, columnspan=2)
        Label(self.createFrame, text="").grid(row=2, column=0, columnspan=2)

        Label(self.createFrame, text="Optimizer").grid(row=2, column=0)
        optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta']
        self.optimizerBox = Combobox(self.createFrame, values=optimizers, state="readonly", width=10)
        self.optimizerBox.set(optimizers[0])
        self.optimizerBox.grid(row=2, column=1)

        self.lossFunctionNames = {
            'Categorical Crossentropy': 'categorical_crossentropy',
            'Sparse Categorical Crossentropy': 'sparse_categorical_crossentropy'
        }
        if allowBinary:
            self.lossFunctionNames["Binary Crossentropy"] = "binary_crossentropy"

        Label(self.createFrame, text="Loss function").grid(row=3, column=0)
        lossFunctions = list(self.lossFunctionNames.keys())
        self.lossFunctionBox = Combobox(self.createFrame, values=lossFunctions,
                                        state="readonly", width=30)
        self.lossFunctionBox.set(lossFunctions[0])
        self.lossFunctionBox.grid(row=3, column=1)

        self.createButton = Button(self.createFrame, text="Create new", command=self.createNew)
        self.createButton.grid(row=4, column=0, columnspan=2)

        # Load
        self.loadFrame = Frame(self.constructionFrame)
        Options.instance.addFrame(self.loadFrame)
        self.loadFrame.grid(row=1, column=1)
        self.loadFrame.columnconfigure(0, weight=1)
        self.loadFrame.columnconfigure(1, weight=1)

        Label(self.loadFrame, text="Load model from file").grid(row=0, column=0)
        Label(self.loadFrame, text="").grid(row=1, column=0, columnspan=2)
        self.loadButton = Button(self.loadFrame, text="Load from file", command=self.loadFromFile)
        self.loadButton.grid(row=4, column=0)
        self.loadButton.config(state="disabled")
        self.foundLabel = Label(self.loadFrame, text="No saved model was found", foreground="red")
        self.checkSavedModel()
        self.foundLabel.grid(row=3, column=0)

    def checkSavedModel(self):
        if os.path.exists(self.modelFilePath):
            self.loadButton.config(state="normal")
            size = format(os.path.getsize(self.modelFilePath) / 1024 / 1024, '.2f') + " MB"
            self.foundLabel.config(text="Saved model available (" + size + ")", foreground="green")

    def getFrame(self):
        return self.constructionFrame

    def loadFromFile(self):
        def load():
            model = tensorflow.keras.models.load_model(self.modelFilePath)
            self.loadButton.config(state="normal", text="Load from file")
            self.loadModel(model)

        self.loadButton.config(state="disabled", text="Loading...")
        threading.Thread(target=load).start()

    def createNew(self):
        def create():
            model = self.createModel()
            metrics = None
            if self.lossFunctionBox.get() == "Binary Crossentropy":
                metrics = [
                    tf.keras.metrics.BinaryCrossentropy(name='Accuracy'),
                ]
            elif self.lossFunctionBox.get() == "Categorical Crossentropy":
                metrics = [
                    tf.keras.metrics.CategoricalAccuracy(name='Accuracy'),
                    tf.keras.metrics.AUC(name='auc')
                ]
            elif self.lossFunctionBox.get() == "Sparse Categorical Crossentropy":
                metrics = [
                    tf.keras.metrics.SparseCategoricalAccuracy(name='Accuracy')
                ]
            model.compile(
                optimizer=self.optimizerBox.get(),
                loss=self.lossFunctionNames[self.lossFunctionBox.get()],
                metrics=metrics
            )
            self.createButton.config(state="normal", text="Create new")
            self.loadModel(model)

        self.createButton.config(state="disabled", text="Creating...")
        threading.Thread(target=create).start()

    def disableButtons(self):
        self.createButton.config(state="disabled")
        self.loadButton.config(state="disabled")

    def enableButtons(self):
        self.createButton.config(state="normal")
        self.loadButton.config(state="normal" if os.path.exists(self.modelFilePath) else "disabled")
