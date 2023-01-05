import tensorflow as tf
import configparser
from utils.utils_metrics import f1_metrics

parser = configparser.ConfigParser()
parser.read("config.txt")


def vgg16_model() -> tf.keras.Model:
    """
    A tensorflow model with the VGG16 architecture.

    Return:
        model (tf.keras.Model): vgg16 after compile
    """
    height = int(parser["image_shape"]["height"])
    width = int(parser["image_shape"]["width"])
    channel = int(parser["image_shape"]["channel"])

    base_model = tf.keras.applications.VGG16(
        weights=parser["model"]["weights"],
        include_top=bool(parser["model"]["include_top"]),
        input_shape=(height, width, channel),
    )
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
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(float(parser["model"]["dropout_rate"])))
    model.add(tf.keras.layers.Dense(1, activation=parser["model"]["activation"]))

    train_enable = bool(parser["model"]["pre_trained_model_trainable"])
    model.layers[0].trainable = train_enable

    model.compile(
        loss=parser["model"]["loss"],
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(parser["model"]["lr"])),
        metrics=[f1_metrics],
    )

    return model
