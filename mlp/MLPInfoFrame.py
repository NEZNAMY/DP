from tkinter import Frame, Label

from pandas import DataFrame

from options import Options


class MLPInfoFrame:

    def __init__(self, parent: Frame, dataSetDisplayName: str, features: DataFrame, classes: dict):
        self.frame = Frame(parent)
        Options.instance.addFrame(self.frame)
        self.frame.grid(row=0, column=1)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.frame.columnconfigure(2, weight=1)
        Label(self.frame, text=dataSetDisplayName, font=("Helvetica", 32)).grid(row=0, column=0, columnspan=99)
        Label(self.frame, font=("Helvetica", 32)).grid(row=1, column=0, columnspan=99)

        self.loadFeatures(features)
        self.loadClasses(classes)

    def loadFeatures(self, features: DataFrame):
        self.featureFrame = Frame(self.frame)
        Options.instance.addFrame(self.featureFrame)
        self.featureFrame.grid(row=2, column=0, sticky="n")
        Label(self.featureFrame, text="Features", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=99)
        Label(self.featureFrame, text="Feature name", font=("Helvetica", 13)).grid(row=1, column=0)
        Label(self.featureFrame, text="Data type", font=("Helvetica", 13)).grid(row=1, column=1)
        Label(self.featureFrame, text="Values", font=("Helvetica", 13)).grid(row=1, column=2)

        string_values_dict = {}
        range_values_dict = {}

        for column in features.columns:
            if features[column].dtype == 'object':
                string_values_dict[column] = features[column].fillna('-').unique().tolist()
            else:
                range_values_dict[column] = [features[column].min(), features[column].max()]

        i = 2
        for name, values in string_values_dict.items():
            Label(self.featureFrame, text=name).grid(row=i, column=0)
            Label(self.featureFrame, text="String", foreground="blue").grid(row=i, column=1)
            Label(self.featureFrame, text=', '.join(sorted(values))).grid(row=i, column=2)
            i += 1

        for name, valueRange in range_values_dict.items():
            Label(self.featureFrame, text=name).grid(row=i, column=0)
            Label(self.featureFrame, text="Number", foreground="orange").grid(row=i, column=1)
            Label(self.featureFrame, text=str(str(valueRange[0]) + " - " + str(valueRange[1]))).grid(row=i, column=2)
            i += 1

    def loadClasses(self, classes: dict):
        self.classesFrame = Frame(self.frame)
        Options.instance.addFrame(self.classesFrame)
        self.classesFrame.grid(row=2, column=1, sticky="n")

        Label(self.classesFrame, text="Classes", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=2)
        Label(self.classesFrame, text="Class name", font=("Helvetica", 13)).grid(row=1, column=0)
        Label(self.classesFrame, text="Entry count", font=("Helvetica", 13)).grid(row=1, column=1)
        i = 2
        for name, count in classes.items():
            Label(self.classesFrame, text=name).grid(row=i, column=0)
            Label(self.classesFrame, text=count, foreground="green").grid(row=i, column=1)
            i += 1

    def getFrame(self):
        return self.frame
