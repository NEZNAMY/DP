from cnn.CNNDataSet import *
from lstm.LSTMDataSet import LSTMDataSet
from mlp import MLPConstructionMenu
from mlp.MLPDataSet import MLPDataSet


class DatasetMenu:

    def __init__(self, tk: Tk, menuBar: Menu):
        self.currentFrame = None
        self.menu = Menu(menuBar, tearoff=0)
        base_dir = 'datasety'

        dataset1 = CNNDataSet(tk, os.path.join(base_dir, 'Dataset_1_Parkinson_Drawing'),
                              "Dataset 1 - Parkinson drawing")
        dataset2 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_2_Parkinson_Speech'),
                              "Dataset 2 - Parkinson Speech")
        dataset3 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_3_Alzheimer_Handwriting'),
                              "Dataset 3 - Alzheimer Handwriting")
        dataset4 = MLPDataSet(tk, os.path.join(base_dir, 'Dataset_4_Finger_Tapping'),
                              "Dataset 4 - Finger Tapping")
        dataset4_2 = LSTMDataSet(tk, os.path.join(base_dir, 'Dataset_4_Finger_Tapping'),
                                 "Dataset 4 - Finger Tapping")
        self.addDataSet(dataset1, "Dataset 1 - Parkinson drawing [CNN]")
        self.menu.add_separator()
        self.addMLPDataSet(dataset2, "Dataset 2 - Parkinson Speech [MLP]")
        self.addMLPDataSet(dataset3, "Dataset 3 - Alzheimer Handwriting [MLP]")
        self.addMLPDataSet(dataset4, "Dataset 4 - Finger Tapping [MLP]")
        self.menu.add_separator()
        self.addDataSet(dataset4_2, "Dataset 4 - Finger Tapping [LSTM]")

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
