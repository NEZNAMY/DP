from tkinter import Frame

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from mlp import MLPDataSet
from shared import AbstractNetwork
from shared.AbstractNetworkFrame import AbstractNetworkFrame


class MLPFrame(AbstractNetworkFrame):

    def __init__(self, parentFrame: Frame, dataSet: MLPDataSet, network: AbstractNetwork):
        super().__init__(parentFrame, dataSet.fullPath, network, dataSet.classCount == 2)
        self.dataSet = dataSet

    def train(self):
        self.trainingFrame.setTrainButtonText("Starting...")

        X_train, X_test, y_train, y_test = self.prepareData()

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(self.modelFilePath, save_best_only=True)
        custom_callback = CustomMLPCallback(self, self.trainingFrame.getEpochCount())

        self.trainingFrame.setTrainButtonText("Training... (0%)")
        self.model.fit(
            X_train,
            y_train,
            epochs=self.trainingFrame.getEpochCount(),
            batch_size=self.trainingFrame.getBatchSize(),
            validation_split=0.2,
            callbacks=[checkpoint_cb, custom_callback]
        )

        self.trainingFrame.setTrainButtonText("Testing results...")
        self.modelInfoFrame.setTrainAccuracy(self.model.evaluate(X_train, y_train)[1])
        self.modelInfoFrame.setTestAccuracy(self.model.evaluate(X_test, y_test)[1])

    def testAccuracy(self):
        X_train, X_test, y_train, y_test = self.prepareData()
        train_results = self.model.evaluate(X_train, y_train)
        test_results = self.model.evaluate(X_test, y_test)
        return [train_results[1], test_results[1]]

    def prepareData(self):
        self.dataSet.features.fillna(0, inplace=True)  # Fill missing values with 0

        X = self.dataSet.features.values  # Convert to NumPy array
        y = self.dataSet.target.values.flatten()

        # Convert string labels to numerical labels using LabelEncoder for target column
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Convert numerical labels to one-hot encoding
        if self.modelInfoFrame.getLossFunctionName(self.model.loss) == "categorical_crossentropy":
            y_encoded = to_categorical(y_encoded)

        # One-hot encode categorical columns in features
        categorical_columns = [i for i, dtype in enumerate(self.dataSet.features.dtypes) if dtype == 'object']
        for col in categorical_columns:
            label_encoder = LabelEncoder()
            X[:, col] = label_encoder.fit_transform(X[:, col].astype(str))

        X = X.astype(float)

        # Scale numerical columns in features
        scaler = StandardScaler()
        numerical_columns = [i for i in range(X.shape[1]) if i not in categorical_columns]
        for col in numerical_columns:
            X[:, col] = scaler.fit_transform(X[:, col].reshape(-1, 1)).flatten()

        # Split the data into training and testing sets
        return train_test_split(X, y_encoded, test_size=0.2, random_state=42)


class CustomMLPCallback(tf.keras.callbacks.Callback):

    def __init__(self, anf: AbstractNetworkFrame, totalEpochs: int):
        super(CustomMLPCallback, self).__init__()
        self.anf = anf
        self.totalEpochs = totalEpochs

    def on_epoch_end(self, epoch, logs=None):
        percent = int((epoch+1) / self.totalEpochs * 100)
        self.anf.trainingFrame.setTrainButtonText("Training... (" + str(percent) + "%)")
