import os
from tkinter import Frame

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

from lstm import LSTMDataSet
from shared import AbstractNetwork
from shared.AbstractNetworkFrame import AbstractNetworkFrame


class LSTMFrame(AbstractNetworkFrame):

    def __init__(self, parentFrame: Frame, dataSet: LSTMDataSet, network: AbstractNetwork):
        super().__init__(parentFrame, dataSet.fullPath, network, len(dataSet.classes))
        self.dataSet = dataSet

    def train(self):
        self.model.getModel().fit(
            self.dataSet.dataX,
            self.dataSet.dataY,
            epochs=self.trainingFrame.getEpochCount(),
            batch_size=self.trainingFrame.getBatchSize(),
        )

    def testAccuracy(self):
        data, labels = self.prepareData()
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        train_results = self.model.getModel().evaluate(x_train, y_train, verbose=0)
        test_results = self.model.getModel().evaluate(x_test, y_test, verbose=0)
        return [train_results[1], test_results[1]]

    def createConfusionMatrix(self):
        X, y_encoded = self.prepareData()
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)

        if self.model.getLossFunction() == "Categorical Crossentropy":
            y_pred = np.argmax(self.model.getModel().predict(X_test_scaled), axis=1)
            cm = confusion_matrix(y_test.argmax(axis=1), y_pred, labels=np.unique(y_encoded.argmax(axis=1)))
        elif self.model.getLossFunction() == "Sparse Categorical Crossentropy":
            y_pred = np.argmax(self.model.getModel().predict(X_test_scaled), axis=1)
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_encoded))
        elif self.model.getLossFunction() == "Binary Crossentropy":
            y_pred_binary = (self.model.getModel().predict(X_test_scaled) > 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred_binary)
        else:
            raise ValueError("Unsupported loss function")
        return ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.dataSet.categories)

    def prepareData(self):
        data = []
        labels = []
        for folder in os.listdir(self.dataSet.fullPath):
            folder_full_path = os.path.join(self.dataSet.fullPath, folder)
            if os.path.isdir(folder_full_path):
                for filename in os.listdir(folder_full_path):
                    filepath = os.path.join(folder_full_path, filename)
                    with open(filepath, 'r') as file:
                        lines = file.readlines()
                    measurement = []
                    for line in lines:
                        values = [float(val.strip()) for val in line.split(',')[1:]]  # Exclude the first column (time)
                        measurement.append(values)
                    measurement = pad_sequences([measurement], maxlen=1997, dtype='float32', padding='post')[0]
                    data.append(measurement)
                    labels.append(folder)  # Use folder name as label
        return np.array(data), LabelEncoder().fit_transform(np.array(labels))


class CustomLSTMCallback(tf.keras.callbacks.Callback):

    def __init__(self, anf: AbstractNetworkFrame, totalEpochs: int):
        super(CustomLSTMCallback, self).__init__()
        self.anf = anf
        self.totalEpochs = totalEpochs

    def on_epoch_end(self, epoch, logs=None):
        percent = int((epoch + 1) / self.totalEpochs * 100)
        self.anf.trainingFrame.setTrainingPhaseText(percent)
