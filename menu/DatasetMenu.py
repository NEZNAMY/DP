import time

from cnn.CNNDataSet import *
from lstm.LSTMDataSet import LSTMDataSet
from mlp import MLPConstructionMenu
from mlp.MLPDataSet import MLPDataSet


class DatasetMenu:

    def __init__(self, tk: Tk, menuBar: Menu):
        self.currentFrame = None
        self.menu = Menu(menuBar, tearoff=0)
        base_dir = 'C:\\Users\\marti\\OneDrive\\Počítač\\škola\\DP\\datasety'

        start_time = time.time_ns()
        dataset1 = CNNDataSet(tk, os.path.join(base_dir, 'Dataset_1_Parkinson_Spiral'),
                              "Dataset 1 - Parkinson spiral drawing")
        print("Loaded dataset 1 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset2 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_2_Alzheimer_Features'),
                              "Dataset 2 - Alzheimer Features")
        print("Loaded dataset 2 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset3 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_3_Alzheimer_Handwriting'),
                              "Dataset 3 - Alzheimer Handwriting")
        print("Loaded dataset 3 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset4 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_4_Alzheimer_Plasma'),
                              "Dataset 4 - Alzheimer Plasma lipidomics")
        print("Loaded dataset 4 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset5 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_5_Parkinson_Features'),
                              "Dataset 5 - Parkinson Features")
        print("Loaded dataset 5 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset6 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_6_Parkinson_Speech'),
                              "Dataset 6 - Parkinson Speech")
        print("Loaded dataset 6 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset99 = LSTMDataSet(tk, os.path.join(base_dir, 'Dataset_99_Test'), "Dataset 99 - Test")
        print("Loaded dataset 99 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        self.addDataSet(dataset1, "Dataset 1 - Parkinson spiral drawing [CNN]")
        self.menu.add_separator()
        self.addMLPDataSet(dataset2, "Dataset 2 - Alzheimer Features [MLP]")
        self.addMLPDataSet(dataset3, "Dataset 3 - Alzheimer Handwriting [MLP]")
        self.addMLPDataSet(dataset4, "Dataset 4 - Alzheimer Plasma [MLP]")
        self.addMLPDataSet(dataset5, "Dataset 5 - Parkinson Features [MLP]")
        self.addMLPDataSet(dataset6, "Dataset 6 - Parkinson Speech [MLP]")
        self.menu.add_separator()
        self.addDataSet(dataset99, "Dataset 99 - Test [LSTM]")

    def addMLPDataSet(self, dataSet: MLPDataSet, displayName: str):
        self.addDataSet(dataSet, displayName)
        MLPConstructionMenu.mlpDataSets.append(dataSet)

    def addDataSet(self, dataSet: AbstractDataSet, displayName: str):
        self.menu.add_command(label=displayName, command=lambda: self.showFrame(dataSet.getFrame()))

    def showFrame(self, frame: Frame):
        if self.currentFrame is not None:
            self.currentFrame.pack_forget()
        self.currentFrame = frame
        frame.pack(expand=True, fill="both")

    def get(self):
        return self.menu

    def name(self):
        return "Dataset"
