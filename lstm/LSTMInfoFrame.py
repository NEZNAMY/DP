from tkinter import Frame, Label

from options import Options


class LSTMInfoFrame:

    def __init__(self, parent: Frame, fullPath: str, dataSetDisplayName: str, classes: dict):
        self.fullPath = fullPath
        self.frame = Frame(parent)
        Options.instance.addFrame(self.frame)
        self.frame.grid(row=0, column=1)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.frame.columnconfigure(2, weight=1)
        Label(self.frame, text=dataSetDisplayName, font=("Helvetica", 32)).grid(row=0, column=0, columnspan=99)
        Label(self.frame, font=("Helvetica", 32)).grid(row=1, column=0, columnspan=99)

        self.loadSignals()
        self.loadClasses(classes)

    def loadSignals(self):
        # TODO
        pass

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
