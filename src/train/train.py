"""Train model"""

import os
import time
import configparser
from keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

parser = configparser.ConfigParser()
parser.read("config.txt")


def train_model(model, train, validation) -> None:
    """
    Train ``model``.

    Args:
        model (tf.keras.Model): model after compile
        train (tf.keras.preprocessing.image.DirectoryIterator)
        validation (tf.keras.preprocessing.image.DirectoryIterator)
    """
    train = train.prefetch(buffer_size=int(parser["train"]["buffer_size"]))
    validation = validation.prefetch(buffer_size=int(parser["train"]["buffer_size"]))

    # Create architecture to save csv_logger
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_output_path = parser["train"]["model_path"] + "results_" + timestamp
    print(f"Getting {model_output_path}")
    if not os.path.isdir(model_output_path):
        os.mkdir(model_output_path)

    custom_callbacks = [
        EarlyStopping(
            monitor=parser["train"]["early_stopping_monitor"],
            mode=parser["train"]["early_stopping_mode"],
            patience=int(parser["train"]["early_stopping_patience"]),
            verbose=int(parser["train"]["early_stopping_verbose"]),
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_output_path + "ckpt", "best_model.h5"),
            monitor=parser["train"]["model_checkpoint_monitor"],
            mode=parser["train"]["model_checkpoint_mode"],
            verbose=int(parser["train"]["model_checkpoint_verbose"]),
            save_best_only=bool(parser["train"]["model_checkpoint_save_best_only"]),
        ),
        CSVLogger(model_output_path + "/training.log", separator=",", append=False),
    ]

    print(f"Saving best model to {model_output_path}")

    model.fit(
        train,
        validation_data=validation,
        epochs=int(parser["train"]["epochs"]),
        callbacks=custom_callbacks,
        validation_steps=len(validation),
        class_weight={
            0: float(parser["train"]["class_0_weight"]),
            1: float(parser["train"]["class_1_weight"]),
        },
    )
    print("End of training")
    return None
