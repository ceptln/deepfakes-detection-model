import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import configparser
from utils.utils_metrics import f1_metrics

parser = configparser.ConfigParser()
parser.read("config.txt")


def resnet_model() -> tf.keras.Model:
    """
    A tensorflow model with the ResNet50 architecture.

    Return:
        model (tf.keras.Model): resnet after compile
    """
    height = int(parser["image_shape"]["height"])
    width = int(parser["image_shape"]["width"])
    channel = int(parser["image_shape"]["channel"])

    base_model = ResNet50(
        weights=parser["model"]["weights"],
        include_top=bool(parser["model"]["include_top"]),
        input_shape=(height, width, channel),
        pooling=parser["model"]["pooling"],
    )

    base_model.trainable = bool(parser["model"]["pre_trained_model_trainable"])

    # Data augmentation to reduce overfitting
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                parser["model"]["random_flip"], input_shape=(height, width, channel)
            ),
            tf.keras.layers.experimental.preprocessing.RandomRotation(
                float(parser["model"]["random_rotation"])
            ),
            tf.keras.layers.experimental.preprocessing.RandomZoom(
                float(parser["model"]["random_zoom"])
            ),
        ]
    )

    model = tf.keras.models.Sequential([data_augmentation])
    model.add(base_model)
    model.add(
        tf.keras.layers.Dense(
            units=int(parser["model"]["dense_unit_layer_1"]),
            activation=parser["model"]["intermediate_activation"],
        )
    )
    model.add(tf.keras.layers.Dropout(float(parser["model"]["dropout_rate"])))
    model.add(
        tf.keras.layers.Dense(
            units=int(parser["model"]["dense_unit_layer_2"]),
            activation=parser["model"]["intermediate_activation"],
        )
    )
    model.add(tf.keras.layers.Dense(1, parser["model"]["activation"]))

    model.compile(
        loss=parser["model"]["loss"],
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(parser["model"]["lr"]),
            #decay=float(parser["model"]["decay"]),
        ),
        metrics=[f1_metrics],
    )

    return model
