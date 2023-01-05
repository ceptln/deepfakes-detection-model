""" Test the loaders for train/validation and test images"""

from src.train.data_generator import train_val_split_generator, image_test_generator
import configparser
import os

parser = configparser.ConfigParser()
parser.read("tests/config_test.txt")


def test_nb_classes_train_val_split_generator():
    """
    Check the number of classes in the train and validation dataset.
    """

    train_data_dir = parser["unit_test_train_data_generator"]["train_path"]
    train_ds, val_ds = train_val_split_generator(train_data_dir)
    nb_classes = len(list(train_ds.as_numpy_iterator()))
    awaited_result = int(
        parser["unit_test_train_data_generator"]["nb_classes_val_data"]
    )
    assert nb_classes == awaited_result


def test_size_image_test_generator():
    """
    Check size of test images.
    """

    height = int(parser["unit_test_train_data_generator"]["height"])
    width = int(parser["unit_test_train_data_generator"]["width"])
    channel = int(parser["unit_test_train_data_generator"]["channel"])
    test_data_dir = parser["unit_test_train_data_generator"]["test_path"]
    test_images = image_test_generator(test_data_dir)
    num_images = len(list(os.listdir(test_data_dir)))
    awaited_result = [num_images, (height, width, channel)]
    assert [len(test_images[0]), test_images[0][0].shape] == awaited_result


def test_number_labels_image_test_generator():
    """
    Check the number of labels of test images.
    """

    test_data_dir = parser["unit_test_train_data_generator"]["test_path"]
    test_images = image_test_generator(test_data_dir)
    awaited_result = len(list(os.listdir(test_data_dir)))
    assert len(test_images[1]) == awaited_result
