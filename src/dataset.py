import tensorflow as tf
import numpy as np
from utils import Datasets
from datasets import load_dataset
import cv2
from sklearn.model_selection import train_test_split
#import tensorflow_datasets as tfds

class DatasetManager:
    def __init__(self):
        self.datasets = {
            Datasets.MNIST.value: self._load_mnist_data(),
            Datasets.FashionMNIST.value: self._load_fashion_data(),
            #Datasets.KMNIST.value: self._load_kmnist_data(),
            #Datasets.EMNIST.value: self._load_emnist_data(),
            #Datasets.CIFAR10.value: self._load_cifar10_data(),
            #Datasets.CIFAR100.value: self._load_cifar100_data(),
            #Datasets.DeepWeeds.value: self._load_deepweeds_data(),
            #Datasets.CitrusLeaves.value: self._load_citrus_leaves_data(),
            #Datasets.SVHN.value: self._load_svhn_data(),
        }

    @staticmethod
    def _split_test_val(x_test, y_test, val_ratio=0.2):
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=val_ratio, stratify=y_test.argmax(axis=1))
        return x_test, y_test, x_val, y_val

    @staticmethod
    def _load_mnist_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis] / 255.0
        x_test = x_test[..., np.newaxis] / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        x_test, y_test, x_val, y_val = DatasetManager._split_test_val(x_test, y_test)
        return x_train, y_train, x_test, y_test, x_val, y_val

    @staticmethod
    def _load_fashion_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train[..., np.newaxis] / 255.0
        x_test = x_test[..., np.newaxis] / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        x_test, y_test, x_val, y_val = DatasetManager._split_test_val(x_test, y_test)
        return x_train, y_train, x_test, y_test, x_val, y_val
    
    @staticmethod
    def _load_kmnist_data():
        ds_train = tfds.load("kmnist", split="train", as_supervised=True)
        ds_test = tfds.load("kmnist", split="test", as_supervised=True)
        ds_train_np = tfds.as_numpy(ds_train)
        ds_test_np = tfds.as_numpy(ds_test)
        x_train, y_train = zip(*list(ds_train_np))
        x_test, y_test = zip(*list(ds_test_np))
        x_train = np.array(x_train).astype("float32") / 255.0
        x_test = np.array(x_test).astype("float32") / 255.0
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        x_test, y_test, x_val, y_val = DatasetManager._split_test_val(x_test, y_test)
        return x_train, y_train, x_test, y_test, x_val, y_val
    
    @staticmethod
    def _load_emnist_data():
        ds_train = tfds.load("emnist/balanced", split="train", as_supervised=True)
        ds_test = tfds.load("emnist/balanced", split="test", as_supervised=True)
        ds_train_np = tfds.as_numpy(ds_train)
        ds_test_np = tfds.as_numpy(ds_test)
        x_train, y_train = zip(*list(ds_train_np))
        x_test, y_test = zip(*list(ds_test_np))
        x_train = np.array(x_train).astype("float32") / 255.0
        x_test = np.array(x_test).astype("float32") / 255.0
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_train = tf.keras.utils.to_categorical(y_train, 47)
        y_test = tf.keras.utils.to_categorical(y_test, 47)
        x_test, y_test, x_val, y_val = DatasetManager._split_test_val(x_test, y_test)
        return x_train, y_train, x_test, y_test, x_val, y_val

    @staticmethod
    def _load_cifar10_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        x_test, y_test, x_val, y_val = DatasetManager._split_test_val(x_test, y_test)
        return x_train, y_train, x_test, y_test, x_val, y_val

    @staticmethod
    def _load_cifar100_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 100)
        y_test = tf.keras.utils.to_categorical(y_test, 100)
        x_test, y_test, x_val, y_val = DatasetManager._split_test_val(x_test, y_test)
        return x_train, y_train, x_test, y_test, x_val, y_val
    
    @staticmethod
    def _load_deepweeds_data():
        ds = tfds.load("deep_weeds", split="train", as_supervised=True)
        ds_np = tfds.as_numpy(ds)
        images, labels = zip(*list(ds_np))
        x = np.array(images)
        y = np.array(labels)
        x = tf.image.resize(x, (64, 64)).numpy()
        x = x.astype("float32") / 255.0
        y = tf.keras.utils.to_categorical(y, 9)
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, stratify=y.argmax(axis=1))
        x_test, y_test, x_val, y_val = DatasetManager._split_test_val(x_temp, y_temp, val_ratio=0.2)
        return x_train, y_train, x_test, y_test, x_val, y_val

    @staticmethod
    def _load_citrus_leaves_data():
        ds = tfds.load("citrus_leaves", split="train", as_supervised=True)
        ds_np = tfds.as_numpy(ds)
        images, labels = zip(*list(ds_np))
        x = np.array(images)
        y = np.array(labels)
        x = tf.image.resize(x, (64, 64)).numpy()
        x = x.astype("float32") / 255.0
        y = tf.keras.utils.to_categorical(y, 4)
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, stratify=y.argmax(axis=1))
        x_test, y_test, x_val, y_val = DatasetManager._split_test_val(x_temp, y_temp, val_ratio=0.2)
        return x_train, y_train, x_test, y_test, x_val, y_val

    @staticmethod
    def _load_svhn_data():
        ds_train = tfds.load("svhn_cropped", split="train", as_supervised=True)
        ds_test = tfds.load("svhn_cropped", split="test", as_supervised=True)
        ds_train_np = tfds.as_numpy(ds_train)
        ds_test_np = tfds.as_numpy(ds_test)
        x_train, y_train = zip(*list(ds_train_np))
        x_test, y_test = zip(*list(ds_test_np))
        x_train = np.array(x_train).astype("float32") / 255.0
        x_test = np.array(x_test).astype("float32") / 255.0
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        x_irrelevant, y_irrelevant, x_remaining, y_remaining = DatasetManager._split_test_val(x_test, y_test, val_ratio=16032/26032)
        x_test, y_test, x_val, y_val = DatasetManager._split_test_val(x_remaining, y_remaining, val_ratio=5200/16032)
        return x_train, y_train, x_test, y_test, x_val, y_val

    def get_dataset(self, name: Datasets):
        return self.datasets.get(name.value)

    def print_shape(self, name: Datasets):
        dataset = self.datasets.get(name.value)
        if dataset is not None:
            x_train, y_train, x_test, y_test, x_val, y_val = dataset
            print(f"Dataset: {name}")
            print(f"x_train shape: {x_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"x_test shape: {x_test.shape}")
            print(f"y_test shape: {y_test.shape}")
            print(f"x_val shape: {x_val.shape}")
            print(f"y_val shape: {y_val.shape}")
        else:
            print(f"Dataset {name} not found.")