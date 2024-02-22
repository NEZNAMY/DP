import math
import os
import random
import shutil
from tkinter import *

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cnn.ModelTestingFrame import ModelTestingFrame
from cnn.types.AbstractCNN import AbstractCNN
from shared.AbstractNetworkFrame import AbstractNetworkFrame


class CNNFrame(AbstractNetworkFrame):

    def __init__(self, parentFrame: Frame, fullPath: str, classWeights: dict,
                 classIndexMapping: dict, network: AbstractCNN):
        super().__init__(parentFrame, fullPath, network, len(classIndexMapping))
        self.classWeights = classWeights
        self.classIndexMapping = classIndexMapping

        self.testingFrame = ModelTestingFrame(self.frame, classIndexMapping)
        self.testingFrame.getFrame().grid(row=2, column=2, sticky="wn", pady=15)

    def loadModel(self, model):
        super().loadModel(model)
        self.testingFrame.loadModel(model, self.network.getImageSize())

    def train(self):

        test_dir = os.path.join(self.fullPath, 'test')
        train_dir = os.path.join(self.fullPath, 'train')

        if (not os.path.exists(train_dir)) or (not os.path.exists(test_dir)):
            self.trainingFrame.setTrainButtonText("Splitting data...")
            self.splitData()

        self.trainingFrame.setTrainButtonText("Starting...")
        count = 0
        for subdir in sorted(os.listdir(train_dir)):
            sub = os.path.join(train_dir, subdir)
            if os.path.isdir(sub):
                count += len(os.listdir(sub))

        trainCount = count-int(0.2*count)

        total_batches = math.ceil(trainCount / self.trainingFrame.getBatchSize()) * self.trainingFrame.getEpochCount()

        def exponential_decay(lr0, s):
            def exponential_decay_fn(epoch):
                return lr0 * 0.1 ** (epoch / s)

            return exponential_decay_fn

        # Callbacks
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay(0.01, 20))
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(self.modelFilePath, save_best_only=True)
        custom_callback = CustomCallback(self, total_batches)

        IMAGE_SIZE = [self.network.getImageSize(), self.network.getImageSize()]
        BATCH_SIZE = self.trainingFrame.getBatchSize()
        seed_value = 1337

        classMode = None
        if self.constructionFrame.lossFunctionBox.get() == "Binary Crossentropy":
            classMode = 'binary'
        elif self.constructionFrame.lossFunctionBox.get() == "Categorical Crossentropy":
            classMode = 'categorical'
        elif self.constructionFrame.lossFunctionBox.get() == "Sparse Categorical Crossentropy":
            classMode = 'sparse'

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            zoom_range=[.99, 1.01],
            horizontal_flip=True,
            fill_mode='constant',
            validation_split=0.2,
            data_format='channels_last'
        )

        val_ds = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode=classMode,
            subset='validation',
            seed=seed_value
        )

        train_ds = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode=classMode,
            subset='training',
            seed=seed_value
        )

        self.trainingFrame.setTrainButtonText("Training... (0%)")

        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.trainingFrame.getEpochCount(),
            class_weight=self.classWeights,
            callbacks=[checkpoint_cb, lr_scheduler, custom_callback]
        )

        self.trainingFrame.setTrainButtonText("Testing results...")

        test_ds = train_datagen.flow_from_directory(
            test_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode=classMode,
            subset='validation',
            seed=seed_value
        )

        self.modelInfoFrame.setTrainAccuracy(self.model.evaluate(train_ds)[1])
        self.modelInfoFrame.setTestAccuracy(self.model.evaluate(test_ds)[1])

    def testAccuracy(self):
        return [0, 0]  # TODO

    def splitData(self):
        train_data_dir = os.path.join(self.fullPath, 'train')
        validation_data_dir = os.path.join(self.fullPath, 'test')

        # Delete directories
        shutil.rmtree(train_data_dir, ignore_errors=True)
        shutil.rmtree(validation_data_dir, ignore_errors=True)

        categories = [f for f in os.listdir(self.fullPath) if os.path.isdir(os.path.join(self.fullPath, f))]

        for category in categories:
            category_path = os.path.join(self.fullPath, category)
            images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

            # Shuffle the list of images
            random.shuffle(images)

            # Determine the split ratio (e.g., 80% train, 20% validation)
            split_ratio = 0.8
            split_index = int(len(images) * split_ratio)

            # Assign images to training and validation sets
            train_images = images[:split_index]
            validation_images = images[split_index:]

            # Create directories for training and validation images
            os.makedirs(os.path.join(train_data_dir, category), exist_ok=True)
            os.makedirs(os.path.join(validation_data_dir, category), exist_ok=True)

            # Move images to the corresponding directories
            for image in train_images:
                source_path = os.path.join(category_path, image)
                destination_path = os.path.join(train_data_dir, category, image)
                shutil.copy(source_path, destination_path)

            for image in validation_images:
                source_path = os.path.join(category_path, image)
                destination_path = os.path.join(validation_data_dir, category, image)
                shutil.copy(source_path, destination_path)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, anf: AbstractNetworkFrame, totalBatches: int):
        super(CustomCallback, self).__init__()
        self.anf = anf
        self.totalBatches = totalBatches
        self.batchCount = 0

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        self.batchCount += 1
        percent = int(self.batchCount/self.totalBatches*100)
        self.anf.trainingFrame.setTrainButtonText("Training... (" + str(percent) + "%)")