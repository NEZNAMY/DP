from tkinter import Frame, Button

from options import Options


class NetworkChoiceMenu:

    def __init__(self, parentFrame: Frame):
        self.currentFrame = None
        self.currentButton = None
        self.index = 0
        self.frame = Frame(parentFrame)
        Options.instance.addFrame(self.frame)
        self.frame.grid(row=0, column=0, sticky="nsew")

    def addNetwork(self, name: str, frame: Frame):
        button = Button(self.frame, text=name, height=2,
                        command=lambda: self.showFrame(frame, button))
        self.frame.grid_columnconfigure(self.index, weight=1)  # Make each column expandable
        button.grid(row=0, column=self.index, sticky="nsew")
        self.index += 1

    def showFrame(self, frame: Frame, button: Button):
        if self.currentFrame is not None:
            self.currentFrame.grid_forget()
        if self.currentButton is not None:
            self.currentButton.config(relief="raised", state="normal")
        self.currentFrame = frame
        frame.grid(row=1, column=0, sticky="nsew")
        button.config(relief="sunken", state="disabled")
        self.currentButton = button
