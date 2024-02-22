from tkinter import Toplevel, Entry, Button, Label, Frame
from tkinter.ttk import Combobox
from typing import Callable

from config import Config
from shared import ActivationFunctions


class MLPLayer:

    def __init__(self, frame: Toplevel, name: str, parameter: str, parameterDisplay: str,
                 activation: str, row: int, remove: Callable):
        self.name = name
        self.parameter = parameter
        self.activation = activation
        self.frame = frame
        self.row = row
        self.nameLabel = Label(self.frame, text=name)
        self.displayLabel = Label(self.frame, text=parameterDisplay)
        self.activationLabel = Label(self.frame, text=activation)
        self.removeButton = Button(self.frame, text="Remove", foreground="red", command=lambda: remove(self))

    def forget(self):
        for widget in self.frame.grid_slaves(row=self.row):
            widget.grid_forget()

    def setRow(self, row: int):
        self.row = row

    def grid(self):
        self.nameLabel.grid(row=self.row, column=0)
        self.displayLabel.grid(row=self.row, column=1, columnspan=2)
        self.activationLabel.grid(row=self.row, column=3)
        self.removeButton.grid(row=self.row, column=4)


class MLPConstructionMenu:

    def __init__(self, takenNames: list, addStructure: Callable):
        self.takenNames = takenNames
        self.addStructure = addStructure
        self.layers = []
        self.window = Toplevel()
        self.window.geometry("700x500")
        self.window.title("Add new MLP network")

        for i in range(0, 9):
            self.window.grid_columnconfigure(i, weight=1)

        Label(self.window, text="Layer type", foreground="green").grid(row=0, column=0)
        Label(self.window, text="Layer parameter", foreground="green").grid(row=0, column=1, columnspan=2)
        Label(self.window, text="Activation function", foreground="green").grid(row=0, column=3)

        # Layer parameter
        self.parameterName = Label(self.window, text="Neurons")
        self.parameterName.grid(row=99, column=1, sticky="e")

        self.parameterEntry = Entry(self.window, width=10)
        self.parameterEntry.grid(row=99, column=2, sticky="w")

        # Layer type
        layers = ["Dense", "Dropout"]
        self.layerBox = Combobox(self.window, values=layers, state="readonly", width=10)
        self.layerBox.set(layers[0])
        self.layerBox.grid(row=99, column=0)

        # Activation function
        functions = ActivationFunctions.getFriendlyValues()
        self.functionsBox = Combobox(self.window, values=functions, state="readonly", width=50)
        self.functionsBox.set(functions[0])
        self.functionsBox.grid(row=99, column=3)

        # Add layer
        self.addButton = Button(self.window, text="Add layer", foreground="green", command=self.addLayer)
        self.addButton.grid(row=99, column=4)
        self.invalidLayerWarn = Label(self.window, foreground="red")
        self.invalidLayerWarn.grid(row=100, column=4)

        def onLayerTypeChange(event):
            if self.layerBox.get() == "Dense":
                self.parameterName.config(text="Neurons")
                self.functionsBox.grid(row=99, column=3)
            elif self.layerBox.get() == "Dropout":
                self.parameterName.config(text="Relative dropout (0-1)")
                self.functionsBox.grid_forget()

        self.layerBox.bind("<<ComboboxSelected>>", onLayerTypeChange)

        # Create window
        self.createWindow = Frame(self.window)
        self.createWindow.grid_columnconfigure(0, weight=1)
        self.createWindow.grid_columnconfigure(1, weight=1)
        Label(self.createWindow, text="Model name").grid(row=0, column=0, sticky="e")
        self.structureName = Entry(self.createWindow, width=10)
        self.structureName.grid(row=0, column=1, sticky="w")
        Button(self.createWindow, text="Create new model structure", height=2, command=self.createStructure).grid(
            row=1, column=0, columnspan=99, pady=5)
        self.createWindowError = Label(self.createWindow, foreground="red")
        self.createWindowError.grid(row=2, column=0, columnspan=99)
        self.createWindow.grid(row=101, column=0, columnspan=99)

        Label(self.window).grid(row=97, pady=30)
        Label(self.window, text="Add new layer", foreground="green").grid(row=98, column=0, columnspan=99)
        self.window.mainloop()

    def createStructure(self):
        if len(self.layers) == 0:
            self.createWindowError.config(text="Could not create model: No layers are defined")
            return
        if self.structureName.get() in self.takenNames:
            self.createWindowError.config(text="Could not create model: This name is already taken")
            return
        if self.structureName.get() == "":
            self.createWindowError.config(text="Could not create model: Name cannot be empty")
            return
        structure = {
            "Name": self.structureName.get(),
            "Layers": []
        }
        for layer in self.layers:
            structure["Layers"].append({
                "LayerType": layer.name,
                "Parameter": layer.parameter,
                "ActivationFunction": ActivationFunctions.encode(layer.activation)
            })
        self.window.destroy()
        Config.instance.addMLPStructure(structure)
        self.addStructure(structure)

    def addLayer(self):
        display = self.parameterEntry.get()
        if self.layerBox.get() == "Dense":
            if not display.isdigit():
                self.invalidLayerWarn.config(text="Invalid amount of neurons: " + self.parameterEntry.get())
                return
            display += " Neurons"
        elif self.layerBox.get() == "Dropout":
            try:
                f = float(display)
                if f <= 0 or f >= 1:
                    self.invalidLayerWarn.config(text="Dropout of of range (0-1): " + self.parameterEntry.get())
                    return
            except ValueError:
                self.invalidLayerWarn.config(text="Invalid number for dropout: " + self.parameterEntry.get())
                return
            display += "%"

        self.invalidLayerWarn.config(text="")
        index = len(self.layers)
        layer = MLPLayer(self.window, self.layerBox.get(), self.parameterEntry.get(), display, self.functionsBox.get(),
                         index+1, self.removeLayer)
        self.layers.append(layer)
        layer.grid()

    def removeLayer(self, layer: MLPLayer):
        layer.forget()
        self.layers.remove(layer)
        i = 1
        for remainingLayer in self.layers:
            remainingLayer.forget()
            remainingLayer.setRow(i)
            remainingLayer.grid()
            i += 1
