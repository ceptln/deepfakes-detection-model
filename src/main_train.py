""" Run training """

from train.data_generator import train_val_split_generator
from train.train import train_model
from train.resnet import resnet_model
from train.vgg16 import vgg16_model
import configparser

parser = configparser.ConfigParser()
parser.read("config.txt")

if __name__ == "__main__":

    train_path = parser["train"]["train_path"]
    if parser["train"]["model_choice"] == "resnet":
        model = resnet_model()
    else:
        model = vgg16_model()
    train_images, val_images = train_val_split_generator(train_path)
    train_model(model, train_images, val_images)
