"""Run the loading, resizing and saving of the images."""

import configparser
from data_loader import load_images

parser = configparser.ConfigParser()
parser.read("../config.txt")
if __name__ == "__main__":

    output_path = parser["data_loader_paths"]["output_path"]
    input_path = parser["data_loader_paths"]["input_path"]

    load_images(input_path, output_path)
