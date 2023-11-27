# Imports
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils
import tensorflow_datasets as tfds

# Set random seed for reproducibility
seed = 88888888
np.random.seed(seed)
tf.random.set_seed(seed)

# Function to load and preprocess SVHN dataset
def load_svhn():
    (svhn_train, svhn_test), ds_info = tfds.load(
        'svhn_cropped', split=['train', 'test'], as_supervised=True, with_info=True, batch_size=-1)

    svhn_train_images, svhn_train_labels = tfds.as_numpy(svhn_train)
    svhn_test_images, svhn_test_labels = tfds.as_numpy(svhn_test)

    # Normalize and preprocess images
    x_train = svhn_train_images.astype("float32") / 255.0
    x_test = svhn_test_images.astype("float32") / 255.0
    
    x_train = x_train / np.max(x_train, axis=(1, 2, 3), keepdims=True)
    x_test = x_test / np.max(x_test, axis=(1, 2, 3), keepdims=True)

    # Replace any division-by-zero
    x_train[np.isnan(x_train)] = 0
    x_test[np.isnan(x_test)] = 0

    # One-hot encode labels
    num_classes = ds_info.features['label'].num_classes
    y_train = utils.to_categorical(svhn_train_labels, num_classes)
    y_test = utils.to_categorical(svhn_test_labels, num_classes)

    return (x_train, y_train), (x_test, y_test),num_classes

# Function to load and preprocess CIFAR-10 dataset
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # One-hot encode labels
    num_classes = 10
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test),num_classes

# Function to load and preprocess CIFAR-100 dataset
def load_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Normalize the data
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # One-hot encode labels
    num_classes = 100
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test),num_classes




# Function to apply custom preprocessing and data augmentation based on the dataset
def preprocessing_(dataset_name, x_train, y_train):
    if dataset_name == "SVHN":
        # Custom preprocessing and data augmentation for SVHN
        datagen = ImageDataGenerator(
            rotation_range=8.0,
            zoom_range=[0.95, 1.05],
            height_shift_range=0.10,
            shear_range=0.15
        )
    elif dataset_name == "CIFAR10":
        # Custom preprocessing and data augmentation for CIFAR-10
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=5./32,
            height_shift_range=5./32,
            horizontal_flip=True
        )
    elif dataset_name == "CIFAR100":
        # Custom preprocessing and data augmentation for CIFAR-100
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=5./32,
            height_shift_range=5./32,
            horizontal_flip=True
        )
    else:
        raise ValueError("Unknown dataset")

    # Fit the ImageDataGenerator to the training data
    datagen.fit(x_train,seed=seed, augment=True)

    return datagen

# Function to load and preprocess dataset based on name
def load_and_preprocess_dataset(dataset_name):
    if dataset_name == "SVHN":
        (x_train, y_train), (x_test, y_test), num_classes = load_svhn()
    elif dataset_name == "CIFAR10":
        (x_train, y_train), (x_test, y_test),num_classes = load_cifar10()
    elif dataset_name == "CIFAR100":
        (x_train, y_train), (x_test, y_test),num_classes = load_cifar100()
    else:
        raise ValueError("Unknown dataset")

    # Preprocessing and data augmentation
    datagen = preprocessing_(dataset_name, x_train, y_train)

    return datagen, (x_train, y_train), (x_test, y_test),num_classes

# Usage Example
#datagen, (x_train, y_train), (x_test, y_test), num_classes = load_and_preprocess_dataset(dataset_name)

