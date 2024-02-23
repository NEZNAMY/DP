import sys
from tkinter import Tk, Menu

from menu.DatasetMenu import DatasetMenu
from menu.FileMenu import FileMenu


class MainWindow:

    def __init__(self, tk: Tk):
        tk.title("Návrh štruktúr neurónových sietí vhodných na detekciu ochorení z meraných signálov")
        tk.geometry("1300x600")
        tk.protocol("WM_DELETE_WINDOW", sys.exit)
        menu_bar = Menu(tk)

        file_menu = FileMenu(tk)
        menu_bar.add_cascade(label=file_menu.name(), menu=file_menu.get())

        dataset_menu = DatasetMenu(tk, menu_bar)
        menu_bar.add_cascade(label=dataset_menu.name(), menu=dataset_menu.get())

        tk.config(menu=menu_bar)
        tk.mainloop()
