import os
import tempfile
from tkinter import Frame, Label

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from PIL import ImageTk, Image
from options import Options


class ModelInfoFrame:

    def __init__(self, parentFrame: Frame):
        self.model = None
        self.modelImage = None
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
        self.lossMappings = {
            # Strings from model defined at runtime
            'binary_crossentropy': "Binary Crossentropy",
            'categorical_crossentropy': 'Categorical Crossentropy',
            'sparse_categorical_crossentropy': 'Sparse Categorical Crossentropy',

            # Classes from model loaded from file
            "BinaryCrossentropy": "Binary Crossentropy",
            'CategoricalCrossentropy': 'Categorical Crossentropy',
            'SparseCategoricalCrossentropy': 'Sparse Categorical Crossentropy'
        }
        self.lossLabel = Label(self.frame, text="Loss function")
        self.lossValue = Label(self.frame, text="", foreground="green")
        self.optimizerLabel = Label(self.frame, text="Optimizer")
        self.optimizerValue = Label(self.frame, text="", foreground="green")
        self.trainAccuracyLabel = Label(self.frame, text="Train data accuracy")
        self.trainAccuracyValue = Label(self.frame, text="-", foreground="green")
        self.testAccuracyLabel = Label(self.frame, text="Test data accuracy")
        self.testAccuracyValue = Label(self.frame, text="-", foreground="green")
        self.modelInfoLabel = Label(self.frame)

    def getFrame(self):
        return self.frame

    def setModel(self, model: Sequential, accuracy: list):
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

        self.model = model
        self.updateModelInfo(accuracy)

    def updateModelInfo(self, accuracy: list):
        self.lossValue.config(text=self.lossMappings.get(self.getLossFunctionName(self.model.loss), "Unknown"))
        self.optimizerValue.config(text=self.model.optimizer.__class__.__name__)
        self.setTrainAccuracy(accuracy[0])
        self.setTestAccuracy(accuracy[1])

        temp_file_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        plot_model(
            self.model,
            to_file=temp_file_path,
            show_shapes=True,
            show_layer_names=False,
            rankdir='TB',  # LR = Left to right, TB = Top to bottom
            dpi=60,
        )
        self.modelImage = ImageTk.PhotoImage(Image.open(temp_file_path))
        os.remove(temp_file_path)
        self.modelInfoLabel.config(image=self.modelImage)

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

    def getLossFunctionName(self, lossFunction):
        if type(lossFunction) is str:
            return lossFunction

        try:
            # MLP
            return lossFunction.__name__
        except AttributeError:
            # CNN
            return lossFunction.__class__.__name__
