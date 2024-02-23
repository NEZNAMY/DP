import os
import tempfile

from PIL import ImageTk, Image
from keras import Sequential
from tensorflow.keras.utils import plot_model


class WrappedModel:

    def __init__(self, model: Sequential):
        self.model = model
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
        self.friendlyToClassMode = {
            "Binary Crossentropy": "binary",
            'Categorical Crossentropy': "categorical",
            'Sparse Categorical Crossentropy': "sparse"
        }
        self.modelImage = self.createModelImage()

    def createModelImage(self):
        temp_file_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        plot_model(
            self.model,
            to_file=temp_file_path,
            show_shapes=True,
            show_layer_names=False,
            rankdir='TB',  # LR = Left to right, TB = Top to bottom
            dpi=60,
        )
        modelImage = ImageTk.PhotoImage(Image.open(temp_file_path))
        os.remove(temp_file_path)
        return modelImage

    def getModel(self):
        return self.model

    def getOptimizer(self):
        return self.model.optimizer.__class__.__name__

    def getLossFunction(self):
        return self.lossMappings.get(self.getLossFunctionName(self.model.loss), "Unknown: " + str(self.model.loss))

    def getLossFunctionClassMode(self):
        return self.friendlyToClassMode[self.getLossFunction()]

    def getLossFunctionName(self, lossFunction):
        if type(lossFunction) is str:
            return lossFunction

        try:
            # MLP
            return lossFunction.__name__
        except AttributeError:
            # CNN
            return lossFunction.__class__.__name__

    def getModelImage(self):
        return self.modelImage
