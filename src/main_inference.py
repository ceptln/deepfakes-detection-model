""" Run inference """

from prediction.inference import infer, save
import configparser

parser = configparser.ConfigParser()
parser.read("./config.txt")

if __name__ == "__main__":
    # Run the inference to get predictions for the test set

    model_path = parser["inference"]["model_path"]
    test_input_path = parser["inference"]["test_data_path"]
    output_path = parser["inference"]["output_path"]
    test_input, res = infer(model_path, test_input_path)
    save(test_input, res, output_path)
