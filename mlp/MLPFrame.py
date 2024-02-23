from tkinter import Frame

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

from mlp import MLPDataSet
from shared import AbstractNetwork
from shared.AbstractNetworkFrame import AbstractNetworkFrame


class MLPFrame(AbstractNetworkFrame):

    def __init__(self, parentFrame: Frame, dataSet: MLPDataSet, network: AbstractNetwork):
        super().__init__(parentFrame, dataSet.fullPath, network, dataSet.classCount)
        self.dataSet = dataSet

    def train(self):
        self.trainingFrame.setStartingPhaseText()

        X, y_encoded = self.prepareData()
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(self.modelFilePath, save_best_only=True)
        custom_callback = CustomMLPCallback(self, self.trainingFrame.getEpochCount())

        self.trainingFrame.setTrainingPhaseText(0)
        self.model.fit(
            X_train_scaled,
            y_train,
            epochs=self.trainingFrame.getEpochCount(),
            batch_size=self.trainingFrame.getBatchSize(),
            validation_split=0.2,
            callbacks=[checkpoint_cb, custom_callback],
            verbose=0
        )

        self.trainingFrame.setTestingTrainData()
        self.modelInfoFrame.setTrainAccuracy(self.model.evaluate(X_train, y_train)[1])

        self.trainingFrame.setTestingTestData()
        self.modelInfoFrame.setTestAccuracy(self.model.evaluate(X_test, y_test)[1])

        self.trainingFrame.setCreatingConfusionMatrix()
        self.updateConfusionMatrix()

    def createConfusionMatrix(self):
        X, y_encoded = self.prepareData()
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if self.modelInfoFrame.model.getLossFunctionName(self.model.loss) == "categorical_crossentropy":
            y_pred = np.argmax(self.model.predict(X_test_scaled), axis=1)
            cm = confusion_matrix(y_test.argmax(axis=1), y_pred, labels=np.unique(y_encoded.argmax(axis=1)))
        elif self.modelInfoFrame.model.getLossFunctionName(self.model.loss) == "sparse_categorical_crossentropy":
            y_pred = np.argmax(self.model.predict(X_test_scaled), axis=1)
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_encoded))
        elif self.modelInfoFrame.model.getLossFunctionName(self.model.loss) == "binary_crossentropy":
            y_pred_binary = (self.model.predict(X_test_scaled) > 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred_binary)
        else:
            raise ValueError("Unsupported loss function")
        return ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.dataSet.categories)

    def testAccuracy(self):
        X, y_encoded = self.prepareData()
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        train_results = self.model.evaluate(X_train, y_train, verbose=0)
        test_results = self.model.evaluate(X_test, y_test, verbose=0)
        return [train_results[1], test_results[1]]

    def getLossFunctionName(self, lossFunction):
        if type(lossFunction) is str:
            return lossFunction

        try:
            # MLP
            return lossFunction.__name__
        except AttributeError:
            # CNN
            return lossFunction.__class__.__name__

    def prepareData(self):
        self.dataSet.features.fillna(0, inplace=True)

        # Convert string labels to numerical labels using LabelEncoder for target column
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(self.dataSet.target.values.flatten())

        # Convert numerical labels to one-hot encoding if using categorical_crossentropy
        if self.getLossFunctionName(self.model.loss) == "categorical_crossentropy":
            y_encoded = to_categorical(y_encoded)

        # One-hot encode categorical columns in features
        categorical_columns = [i for i, dtype in enumerate(self.dataSet.features.dtypes) if dtype == 'object']
        X = self.dataSet.features.values.copy()  # Copy to avoid modifying the original DataFrame

        for col in categorical_columns:
            label_encoder = LabelEncoder()
            X[:, col] = label_encoder.fit_transform(X[:, col].astype(str))

        X = X.astype(float)

        # Scale numerical columns in features
        scaler = StandardScaler()
        numerical_columns = [i for i in range(X.shape[1]) if i not in categorical_columns]
        for col in numerical_columns:
            X[:, col] = scaler.fit_transform(X[:, col].reshape(-1, 1)).flatten()

        return X, y_encoded


class CustomMLPCallback(tf.keras.callbacks.Callback):

    def __init__(self, anf: AbstractNetworkFrame, totalEpochs: int):
        super(CustomMLPCallback, self).__init__()
        self.anf = anf
        self.totalEpochs = totalEpochs

    def on_epoch_end(self, epoch, logs=None):
        percent = int((epoch+1) / self.totalEpochs * 100)
        self.anf.trainingFrame.setTrainingPhaseText(percent)
