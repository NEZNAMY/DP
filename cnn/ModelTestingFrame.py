import threading
from tkinter import Frame, Label, Button, filedialog
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from keras import Sequential

from options import Options


class ModelTestingFrame:

    def __init__(self, parentFrame: Frame, classIndexMapping: dict):
        self.model = None
        self.size = ()
        self.classIndexMapping = classIndexMapping
        self.frame = Frame(parentFrame)
        Options.instance.addFrame(self.frame)
        self.frame.columnconfigure(0, weight=1)

        Label(self.frame, text="Testing menu", font=16).grid(row=0, column=0)
        Label(self.frame, text="", font=16).grid(row=1, column=0)

        self.noModelWarn = Label(self.frame, text="Testing is not available: No model is loaded", foreground="red")
        self.noModelWarn.grid(row=2, column=0)

        self.testButton = Button(self.frame, text="Select image to test", command=self.test)

        self.resultLabel = Label(self.frame, text="Image test results with probability")
        self.probabilityLabels = []
        for _ in classIndexMapping.keys():
            self.probabilityLabels.append(Label(self.frame, foreground="green"))

    def getFrame(self):
        return self.frame

    def loadModel(self, model: Sequential, imgSize: tuple):
        if self.model is not None:
            self.hideLabels()
        self.model = model
        self.size = (imgSize, imgSize)
        self.noModelWarn.grid_forget()
        self.testButton.grid(row=2, column=0)

    def test(self):
        def testAsync(path):
            img = image.load_img(path, target_size=self.size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            predictions = self.model.predict(img_array)
            sorted_indices = np.argsort(predictions[0])[::-1]  # Sort indices by probability

            self.showLabels()

            for i in sorted_indices:
                class_name = self.classIndexMapping[i]
                probability = predictions[0][i] * 100
                self.probabilityLabels[i].config(text=class_name + " - " + format(probability, ".2f") + "%")

            self.testButton.config(state="normal", text="Select image to test")

        file_path = filedialog.askopenfilename(title="Select an image to test")
        if file_path == "":  # Cancelled
            return

        self.testButton.config(state="disabled", text="Testing image...")
        self.hideLabels()
        threading.Thread(target=lambda: testAsync(file_path)).start()

    def showLabels(self):
        self.resultLabel.grid(row=3, column=0)
        for i in self.classIndexMapping.keys():
            self.probabilityLabels[i].grid(row=i + 4, column=0)

    def hideLabels(self):
        self.resultLabel.grid_forget()
        for i in self.classIndexMapping.keys():
            self.probabilityLabels[i].grid_forget()
