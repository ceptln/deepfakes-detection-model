"""Test the data loader """

from src.data_processing.data_loader import load_images
from PIL import Image
from tqdm import tqdm
import os
import configparser

parser = configparser.ConfigParser()
parser.read("tests/config_test.txt")


def test_load_images_creation_folders():
    """
    Test the creation of the folders when loading the data.
    """

    input_path = parser["unit_test_data_loader"]["input_path"]
    output_path = parser["unit_test_data_loader"]["output_path"]
    load_images(input_path, output_path)

    awaited_result = [True]
    bools = [
        os.path.isdir(output_path + "images"),
        os.path.isdir(output_path + "images/images_train/0"),
        os.path.isdir(output_path + "images/images_train/1"),
        os.path.isdir(output_path + "/images/images_test"),
    ]
    assert bools == awaited_result * len(bools)


def test_load_images_files_in_folders():
    """
    Test the presence of images in the created folders when loading the data.
    """

    output_path = parser["unit_test_data_loader"]["output_path"]

    awaited_result = [True]
    bools = [
        len(os.listdir(output_path + "images")) != 0,
        len(os.listdir(output_path + "images/images_train/0")) != 0,
        len(os.listdir(output_path + "images/images_train/1")) != 0,
        len(os.listdir(output_path + "images/images_test")) != 0,
    ]
    assert bools == awaited_result * len(bools)


def test_load_images_size_test_folder():
    """
    Test the size of the images in the test folder.
    """

    output_path = parser["unit_test_data_loader"]["output_path"]
    img_height = int(parser["unit_test_data_loader"]["img_height"])
    img_width = int(parser["unit_test_data_loader"]["img_width"])

    awaited_result = [(img_height, img_width)]
    test_folder = output_path + "images/images_test"
    sizes = []
    for i in tqdm(os.listdir(test_folder)):
        image = Image.open(test_folder + "/" + i)
        sizes.append(image.size)
    assert sizes == awaited_result * len(sizes)


def test_load_images_size_train0_folder():
    """
    Test the size of the images in the train0 folder.
    """

    output_path = parser["unit_test_data_loader"]["output_path"]
    img_height = int(parser["unit_test_data_loader"]["img_height"])
    img_width = int(parser["unit_test_data_loader"]["img_width"])

    awaited_result = [(img_height, img_width)]
    train_0_folder = output_path + "images/images_train/0"
    sizes = []
    for i in tqdm(os.listdir(train_0_folder)):
        image = Image.open(train_0_folder + "/" + i)
        sizes.append(image.size)
    assert sizes == awaited_result * len(sizes)


def test_load_images_size_train1_folder():
    """
    Test the size of the images in the train1 folder.
    """

    output_path = parser["unit_test_data_loader"]["output_path"]
    img_height = int(parser["unit_test_data_loader"]["img_height"])
    img_width = int(parser["unit_test_data_loader"]["img_width"])

    awaited_result = [(img_height, img_width)]
    train_1_folder = output_path + "images/images_train/1"
    sizes = []
    for i in tqdm(os.listdir(train_1_folder)):
        image = Image.open(train_1_folder + "/" + i)
        sizes.append(image.size)
    assert sizes == awaited_result * len(sizes)
