from tkinter import *

from config import Config


class Options:

    def __init__(self):
        self.frames = []
        self.window = Toplevel()
        self.window.withdraw()
        self.window.geometry("200x100")
        self.window.protocol("WM_DELETE_WINDOW", self.onClose)
        self.showFrameBorders = BooleanVar(value=Config.instance.isShowFrameBorders())
        showFramesCheckBox = Checkbutton(self.window, text="Show frame borders",
                                         variable=self.showFrameBorders, command=self.onShowFrameBordersChange)
        showFramesCheckBox.pack()

    def onShowFrameBordersChange(self):
        for frame in self.frames:
            frame.config(highlightthickness=1 if self.showFrameBorders.get() else 0)
        Config.instance.setShowFrameBorders(self.showFrameBorders.get())

    def addFrame(self, frame: Frame):
        frame.config(highlightbackground="black", highlightthickness=1 if self.showFrameBorders.get() else 0)
        self.frames.append(frame)

    def onClose(self):
        self.window.withdraw()

    def showMenu(self):
        self.window.deiconify()
        self.window.mainloop()


instance = None
