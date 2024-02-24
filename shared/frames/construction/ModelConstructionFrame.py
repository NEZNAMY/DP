import os
import threading
from tkinter import Label, Button, Frame
from tkinter.ttk import Combobox
from typing import Callable

import tensorflow as tf

from options import Options
from shared import AbstractNetworkFrame
from shared.WrappedModel import WrappedModel
from shared.frames.construction.ModelLoadingFromFile import ModelLoadingFromFile


class ModelConstructionFrame:

    def __init__(self, parent: AbstractNetworkFrame, modelFilePath: str, loadModel: Callable, createModel: Callable,
                 classCount: int):
        self.parent = parent
        self.modelFilePath = modelFilePath
        self.loadModel = loadModel
        self.createModel = createModel
        self.classCount = classCount
        self.constructionFrame = Frame(parent.getFrame())
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
        if classCount == 2:
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
        self.modelLoading = ModelLoadingFromFile(self, self.modelFilePath)
        self.modelLoading.getFrame().grid(row=1, column=1)

    def getFrame(self):
        return self.constructionFrame

    def createNew(self):
        def create():
            outputSize = self.classCount if self.lossFunctionBox.get() != "Binary Crossentropy" else 1
            outputActivation = "softmax" if self.lossFunctionBox.get() != "Binary Crossentropy" else "sigmoid"
            model = self.createModel(outputSize, outputActivation)
            metrics = ["accuracy"]
            if self.lossFunctionBox.get() == "Categorical Crossentropy":
                metrics.append(tf.keras.metrics.AUC(name='auc'))
            model.compile(
                optimizer=self.optimizerBox.get(),
                loss=self.lossFunctionNames[self.lossFunctionBox.get()],
                metrics=metrics
            )
            self.loadModel(WrappedModel(model), self.createButton)
            self.createButton.config(state="normal", text="Create new")

        self.createButton.config(state="disabled", text="Phase 1/3: Constructing model...")
        threading.Thread(target=create).start()

    def disableButtons(self):
        self.createButton.config(state="disabled")
        self.modelLoading.loadButton.config(state="disabled")

    def enableButtons(self):
        self.createButton.config(state="normal")
        self.modelLoading.loadButton.config(state="normal" if os.path.exists(self.modelFilePath) else "disabled")
