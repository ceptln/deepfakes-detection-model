""" Loaders for train/validation and test images """

import os
from typing import Tuple, MutableSequence
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import configparser

parser = configparser.ConfigParser()
parser.read("config.txt")


def train_val_split_generator(
    train_data_dir,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Returns train and validation dataset for the model

    Train validation split using ``image_dataset_from_directory`` from keras
    with featurewise centering and normalization.

    Train images ``train_data_dir`` placed into subdirectories based on label.

    Args:
        train_data_dir (str): the path to the train dataset
        seed (int): random seed for train/validation split

    Returns:
        train_images, val_images (tf.data.Dataset)：train val data sets
    """
    height = int(parser["image_shape"]["height"])
    width = int(parser["image_shape"]["width"])
    validation_split = float(parser["image_data_generator"]["validation_split"])
    seed = int(parser["image_data_generator"]["seed"])
    shuffle = bool(parser["image_data_generator"]["shuffle"])
    batch_size = int(parser["image_data_generator"]["batch_size"])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(height, width),
        shuffle=shuffle,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(height, width),
        shuffle=shuffle,
        batch_size=batch_size,
    )
    return train_ds, val_ds


def image_test_generator(test_data_dir) -> MutableSequence:
    """
    Returns test dataset for evaluating the model

    Read images as test images, the result is a list containing two lists:
       - the ``i`` th element of the first sublist is an image in
         test_data_dir in numpy array format
       - the ``i`` th element of the second sublist
       the image id of the ``i``th element in the first list

    Args:
        test_data_dir (str): the path to the test dataset

    Returns:
        test_images (np.array): all test images in a numpy array
    """
    # convert image to numpy array
    test_images = [[], []]
    height = int(parser["image_shape"]["height"])
    width = int(parser["image_shape"]["width"])
    channel = int(parser["image_shape"]["channel"])
    for i in tqdm(os.listdir(test_data_dir)):
        try:
            image = Image.open(test_data_dir + "/" + i)
            image = np.array(image, dtype=np.uint8)
            if image.shape == (height, width):
                image = np.stack((image,) * channel, axis=-1)
            test_images[0].append(image)
            test_images[1].append(i)
        except:
            pass

    return test_images


if __name__ == "__main__":
    # test module with sample data set

    test_test_data_dir = parser["image_data_generator"]["test_sample_data_dir"]
    test_test = image_test_generator(test_test_data_dir)
    num_images = len(list(os.listdir(test_test_data_dir)))

    height = int(parser["image_shape"]["height"])
    width = int(parser["image_shape"]["width"])
    channel = int(parser["image_shape"]["channel"])

    print("Check size of images:")
    print(f"{test_test[0].shape == (num_images,  height, width, channel)}")
    print(f"Check number of label: {len(test_test[1]) == num_images}")
