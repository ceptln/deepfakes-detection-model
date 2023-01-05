""" Test the inference """

from src.prediction.inference import infer, save
import configparser
import pandas as pd
import os

parser = configparser.ConfigParser()
parser.read("tests/config_test.txt")


def test_size_predictions():
    """
    Test that the correct number of predictions is made.
    """

    model_path = parser["unit_test_inference"]["model_path"]
    test_path = parser["unit_test_inference"]["test_path"]
    test_input, test_res = infer(model_path, test_path)
    size_pred = len(test_res)
    awaited_result = int(parser["unit_test_inference"]["size_pred"])
    assert size_pred == awaited_result


def test_results_saved():
    """
    Test that the results are saved.
    """

    model_path = parser["unit_test_inference"]["model_path"]
    test_path = parser["unit_test_inference"]["test_path"]
    test_input, test_res = infer(model_path, test_path)
    output_path = parser["unit_test_inference"]["output_path"]
    save(test_input, test_res, output_path)
    awaited_result = True
    bool = os.path.isfile(output_path)
    assert bool == awaited_result


def test_probability_results_converted_to_classes():
    """
    Test that the outputs are 0's or 1's.
    """

    results = pd.read_csv(parser["unit_test_inference"]["output_path"], delimiter=";")
    print(results)
    bool = results["label"].isin([0, 1]).all()
    awaited_result = True
    assert bool == awaited_result
