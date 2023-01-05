"""Load images, resize and store them."""

import os
import configparser
from tqdm import tqdm
from yoloface import face_analysis
import pandas as pd
from PIL import Image
import numpy as np

parser = configparser.ConfigParser()
parser.read("../config.txt")


def expand(box, image):
    """
    Return the image cropped around the face detected by yoloface
    Adds a 30% margin to the face if the position of the box allows
    Args:
        box (list): encodes the bounding box of the face in the image
        image (np.array): image encoded as numpy array
    Returns:
        image (np.array): cropped image
    """
    if box == []:
        return image
    x = box[0][0]
    y = box[0][1]
    w = box[0][2]
    h = box[0][3]
    y1 = y - int(w * 0.15)
    y2 = w + y + int(w * 0.15)
    x1 = x - int(h * 0.15)
    x2 = x + h + int(h * 0.15)
    if y1 < 0:
        y1 = 0
    if x1 < 0:
        x1 = 0
    if y2 > image.shape[0]:
        y2 = image.shape[0]
    if x2 > image.shape[1]:
        x2 = image.shape[1]
    return image[y1:y2, x1:x2]


def load_images(input_path, output_path) -> None:
    """
    Load image from folder path and place them into folders for train and test.

    Args:
        input_path (str):
            path for stored input images, train.csv and test.csv.
        output_path (str):
            path where resized images are stored.
    """
    # Create corresponding folders
    if not os.path.isdir(output_path + "images"):
        os.mkdir(output_path + "images")
    if not os.path.isdir(output_path + "images/images_train/0"):
        os.mkdir(output_path + "images/images_train")
        os.mkdir(output_path + "images/images_train/0")
    if not os.path.isdir(output_path + "images/images_train/1"):
        os.mkdir(output_path + "images/images_train/1")
    if not os.path.isdir(output_path + "images/images_test"):
        os.mkdir(output_path + "images/images_test")
    # Read
    testcsv = pd.read_csv(input_path + "test.csv")
    traincsv = pd.read_csv(input_path + "train.csv")
    testcsv["image_id"] = testcsv["image_id"].astype("string")
    traincsv["image_id"] = traincsv["image_id"].astype("string")
    for i in tqdm(os.listdir(input_path + "images")):
        try:
            face_extract = bool(parser["data_processing"]["enable_face_extraction"])
            if face_extract is True:
                face = face_analysis()
                img, box, conf = face.face_detection(
                    image_path=input_path + "images" + "/" + i, model="full"
                )
                image = np.array(
                    Image.open(input_path + "images" + "/" + i), dtype=np.uint8
                )
                image = expand(box, image)
                image = Image.fromarray(image)
                img = image.resize((224, 224))
                if i.split(".jpg")[0] in testcsv.values:
                    img.save(output_path + "images/images_test/" + i)
                elif i.split(".jpg")[0] in traincsv.values[:, 0]:
                    if conf[0] >= float(
                        parser["data_processing"]["face_detection_conf"]
                    ):
                        label = str(
                            traincsv.loc[
                                traincsv["image_id"] == i.split(".jpg")[0], "label"
                            ].iloc[0]
                        )
                        if label == "1":
                            img.save(
                                output_path + "images/images_train/" + label + "/" + i
                            )
                        elif label == "0":
                            img.save(
                                output_path
                                + "images/images_train/"
                                + label
                                + "/"
                                + i.split(".jpg")[0]
                                + ".jpg"
                            )
            else:
                image = np.array(
                    Image.open(input_path + "images" + "/" + i), dtype=np.uint8
                )
                image = Image.fromarray(image)
                img = image.resize((224, 224))
                if i.split(".jpg")[0] in testcsv.values:
                    img.save(output_path + "images/images_test/" + i)
                elif i.split(".jpg")[0] in traincsv.values[:, 0]:
                    if label == "1":
                        img.save(output_path + "images/images_train/" + label + "/" + i)
                    elif label == "0":
                        img.save(
                            output_path
                            + "images/images_train/"
                            + label
                            + "/"
                            + i.split(".jpg")[0]
                            + ".jpg"
                        )
        except:
            pass

    return None
