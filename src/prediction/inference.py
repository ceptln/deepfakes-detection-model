import tensorflow as tf
import os
import numpy as np
import pandas as pd
import configparser
from train.data_generator import image_test_generator
from utils.utils_metrics import f1_metrics
from typing import Tuple, MutableSequence, List

parser = configparser.ConfigParser()
parser.read("./config.txt")


def infer(model_path, test_path) -> Tuple[List, MutableSequence]:
    """
    Infere the labels of images in the test path.

    Infer the label of images in ``test_path`` using
    tensorflow model stored in ``model_path``

    Args:
        model_path (str): the path to a saved tensorflow model
        test_path (str): the path to the folder containing test images (*.jpg)

    Returns:
        test_input (list): output of ``image_test_generator``
        test_res (np.array): probability list each image belonging to class 1
    """
    reconstructed_model = tf.keras.models.load_model(
        os.path.join(model_path, "best_model.h5"),
        custom_objects={"f1_metrics": f1_metrics},
    )

    print("Reading images...")
    test_input = image_test_generator(test_path)

    print("Inferring...")
    test_res = reconstructed_model.predict(
        np.array(test_input[0]), verbose=int(parser["inference"]["verbose"])
    )
    return test_input, test_res


def save(test_input, test_res, output_path) -> None:
    """
    Save the result of inference

    Assign a binary label to each image according to its probability,
    the result is stored in a csv file with one column containing
    the image id and its inferred label.

    Args:
        test_input (list): output of ``image_test_generator``
        test_res (np.array): probability list each image belonging to class 1
        output_path (str): the path to the result csv file
    """

    print("Assigning labels...")
    threshold = float(parser["inference"]["threshold"])
    id_res = pd.DataFrame(
        [
            test_input[1][i].split(".jpg")[0] + ";" + str(int(test_res[i] > threshold))
            for i in range(len(test_res))
        ]
    )
    id_res.columns = ["image_id;label"]

    print("Storing predictions...")
    id_res.to_csv(output_path, index=False)
    return None


if __name__ == "__main__":
    # test module with sample data set

    test_model_path = parser["inference"]["test_model_path"]
    test_input_path = parser["inference"]["test_input_path"]

    input, res = infer(test_model_path, test_input_path)
    save(input, res, "test_result.csv")
