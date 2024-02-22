from abc import ABC, abstractmethod
from tkinter import Frame, Tk

from menu.NetworkChoiceMenu import NetworkChoiceMenu
from menu.Sidebar import Sidebar
from options import Options


class AbstractDataSet(ABC):

    def __init__(self, tk: Tk, fullPath: str, displayName: str):
        self.fullPath = fullPath
        self.displayName = displayName

        self.categories = self.loadCategories()
        self.classIndexMapping = {i: className for i, className in enumerate(self.categories)}

        self.frame = Frame(tk)
        Options.instance.addFrame(self.frame)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)

        self.trainingFrame = Frame(self.frame)
        Options.instance.addFrame(self.trainingFrame)
        self.trainingFrame.grid_columnconfigure(0, weight=1)
        self.trainingFrame.grid_rowconfigure(0, weight=0)
        self.trainingFrame.grid_rowconfigure(1, weight=1)
        self.sidebar = Sidebar(self.frame)
        self.sidebar.addOption("Info", self.createInfoFrame())
        self.sidebar.addOption("Training", self.trainingFrame)

        self.networks = NetworkChoiceMenu(self.trainingFrame)
        self.loadNetworks()

    def getFrame(self):
        return self.frame

    @abstractmethod
    def createInfoFrame(self):
        pass

    @abstractmethod
    def loadNetworks(self):
        pass

    @abstractmethod
    def loadCategories(self):
        pass
