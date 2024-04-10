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
        dataset1 = CNNDataSet(tk, os.path.join(base_dir, 'Dataset_1_Parkinson_Drawing'),
                              "Dataset 1 - Parkinson drawing")
        print("Loaded dataset 1 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset2 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_2_Parkinson_Speech'),
                              "Dataset 2 - Parkinson Speech")
        print("Loaded dataset 2 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset3 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_3_Alzheimer_Handwriting'),
                              "Dataset 3 - Alzheimer Handwriting")
        print("Loaded dataset 3 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset102 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_102_Alzheimer_Features'),
                                "Dataset 102 - Alzheimer Features")
        print("Loaded dataset 102 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset104 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_104_Alzheimer_Plasma'),
                                "Dataset 104 - Alzheimer Plasma lipidomics")
        print("Loaded dataset 104 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset105 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_105_Parkinson_Features'),
                                "Dataset 105 - Parkinson Features")
        print("Loaded dataset 105 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        start_time = time.time_ns()
        dataset99 = LSTMDataSet(tk, os.path.join(base_dir, 'Dataset_99_Test'), "Dataset 99 - Test")
        print("Loaded dataset 99 in " + str(int((time.time_ns() - start_time) / 1000000)) + "ms")

        self.addDataSet(dataset1, "Dataset 1 - Parkinson drawing [CNN]")
        self.menu.add_separator()
        self.addMLPDataSet(dataset2, "Dataset 2 - Parkinson Speech [MLP]")
        self.addMLPDataSet(dataset3, "Dataset 3 - Alzheimer Handwriting [MLP]")
        self.addMLPDataSet(dataset102, "Dataset 102 - Alzheimer Features [MLP]")
        self.addMLPDataSet(dataset104, "Dataset 104 - Alzheimer Plasma [MLP]")
        self.addMLPDataSet(dataset105, "Dataset 105 - Parkinson Features [MLP]")

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
