""" Test the visualization of the metrics. """

from src.visualizations.metrics import plot_loss_f1_score
import pandas as pd
import configparser
import os

parser = configparser.ConfigParser()
parser.read("tests/config_test.txt")


def test_save_figure():
    """
    Test that the figure is saved.
    """

    model_path = parser["unit_test_visualization"]["model_path"]
    filename = parser["unit_test_visualization"]["filename"]
    model = pd.read_csv(model_path + "training.log", sep=",", engine="python")
    plot_loss_f1_score(model, model_path, filename)

    awaited_result = True
    output_path = model_path + filename
    bool = os.path.isfile(output_path)
    assert bool == awaited_result
