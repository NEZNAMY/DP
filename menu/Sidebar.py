from tkinter import *

from options import Options


class Sidebar:

    def __init__(self, parentFrame: Frame):
        self.currentFrame = None
        self.currentButton = None
        self.index = 0
        self.frame = Frame(parentFrame)
        Options.instance.addFrame(self.frame)
        self.frame.grid(row=0, column=0, sticky="nsew")

    def addOption(self, name: str, frame: Frame):
        button = Button(self.frame, text=name, height=3, width=15,
                        command=lambda: self.showFrame(frame, button))
        button.grid(row=self.index, column=0, sticky="ew")
        self.index = self.index + 1
        if self.index == 1:
            self.showFrame(frame, button)

    def showFrame(self, frame: Frame, button: Button):
        if self.currentFrame is not None:
            self.currentFrame.grid_forget()
        if self.currentButton is not None:
            self.currentButton.config(relief="raised", state="normal")
        self.currentFrame = frame
        frame.grid(row=0, column=1, sticky="nsew")
        button.config(relief="sunken", state="disabled")
        self.currentButton = button
