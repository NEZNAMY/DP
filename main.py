from tkinter import Tk

from menu.MainWindow import MainWindow
from options import Options

tk = Tk()
Options.instance = Options.Options()
MainWindow(tk)
