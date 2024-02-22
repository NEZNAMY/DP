from tkinter import Frame, Label

from options import Options


class LSTMInfoFrame:

    def __init__(self, parent: Frame, dataSetDisplayName: str):
        self.frame = Frame(parent)
        Options.instance.addFrame(self.frame)
        self.frame.grid(row=0, column=1)
        self.frame.columnconfigure(0, weight=1)
        Label(self.frame, text=dataSetDisplayName, font=("Helvetica", 32)).grid(row=0, column=0, columnspan=99)
        Label(self.frame, font=("Helvetica", 32)).grid(row=1, column=0, columnspan=99)

    def getFrame(self):
        return self.frame
