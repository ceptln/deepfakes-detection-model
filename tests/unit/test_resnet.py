""" Test the resnet model. """

from src.train.resnet import resnet_model
import configparser
import pandas as pd
import re

parser = configparser.ConfigParser()
parser.read("tests/config_test.txt")


def test_number_layers_resnet():
    """
    Test the number of layers in the resnet model.
    """

    model = resnet_model()
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summ_string = "\n".join(stringlist)
    print(summ_string)  # entire summary in a variable

    table = stringlist[1:-4][1::2]  # take every other element and remove appendix

    new_table = []
    for entry in table:
        entry = re.split(r"\s{2,}", entry)[:-1]  # remove whitespace
        new_table.append(entry)

    df = pd.DataFrame(new_table[1:], columns=new_table[0])
    number_layers = len(df.index) - 1
    awaited_result = int(parser["unit_test_resnet"]["nb_layers"])
    assert number_layers == awaited_result


def test_number_params_resnet():
    """
    Test the number of params in the resnet model.
    """

    model = resnet_model()
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summ_string = "\n".join(stringlist)
    print(summ_string)  # entire summary in a variable

    table = stringlist[1:-4][1::2]  # take every other element and remove appendix

    new_table = []
    for entry in table:
        entry = re.split(r"\s{2,}", entry)[:-1]  # remove whitespace
        new_table.append(entry)

    df = pd.DataFrame(new_table[1:], columns=new_table[0])
    number_params = (
        df["Param #"][: int(parser["unit_test_resnet"]["nb_layers"])]
        .astype("int32")
        .sum()
    )
    awaited_result = int(parser["unit_test_resnet"]["nb_params"])
    assert number_params == awaited_result
