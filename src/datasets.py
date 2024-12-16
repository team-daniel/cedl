import tensorflow as tf
import numpy as np
from utils import Datasets

class DatasetManager:
    def __init__(self):
        self.datasets = {
            Datasets.MNIST.value: self._load_mnist_data(),
            Datasets.FashionMNIST.value: self._load_fashion_data(),
        }

    @staticmethod
    def _load_mnist_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis] / 255.0
        x_test = x_test[..., np.newaxis] / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def _load_fashion_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train[..., np.newaxis] / 255.0
        x_test = x_test[..., np.newaxis] / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        return x_train, y_train, x_test, y_test

    def get_dataset(self, name: Datasets):
        return self.datasets.get(name.value)

    def print_shape(self, name: Datasets):
        dataset = self.datasets.get(name.value)
        if dataset is not None:
            x_train, y_train, x_test, y_test = dataset
            print(f"Dataset: {name}")
            print(f"x_train shape: {x_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"x_test shape: {x_test.shape}")
            print(f"y_test shape: {y_test.shape}")
        else:
            print(f"Dataset {name} not found.")