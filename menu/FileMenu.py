import sys
from tkinter import *

from options import Options


class FileMenu:
    
    def __init__(self, menuBar):
        self.menu = Menu(menuBar, tearoff=0)
        self.menu.add_command(label="Options", command=Options.instance.showMenu)
        self.menu.add_separator()
        self.menu.add_command(label="Exit", command=sys.exit)

    def get(self):
        return self.menu

    def name(self):
        return "File"
