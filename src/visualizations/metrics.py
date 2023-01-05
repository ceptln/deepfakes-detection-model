""" Plot the metrics. """

import matplotlib.pyplot as plt
import pandas as pd
import configparser

parser = configparser.ConfigParser()
parser.read("../config.txt")


def plot_loss_f1_score(
    model, model_path, filename="training_validation_f1_score.png"
) -> None:
    """
    Save each of the Training Validation accuracy and loss in a png file.
    """
    loss = model["loss"]
    val_loss = model["val_loss"]
    f1_score = model["f1_metrics"]
    val_f1_score = model["val_f1_metrics"]
    epochs = range(len(loss))

    # PLot loss and accuracy for tuning
    plt.plot(epochs, loss, "b", label="Training Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.savefig(model_path + "training_validation_loss.png")

    plt.show()
    plt.clf()

    plt.plot(epochs, f1_score, "b", label="Training F1 Score")
    plt.plot(epochs, val_f1_score, "r", label="Validation F1 Score")
    plt.title("Training and Validation F1 Score")
    plt.legend()

    plt.savefig(model_path + filename)

    plt.show()
    return None


if __name__ == "__main__":

    model_path = parser["visualization"]["model_path"]
    log_data = pd.read_csv(model_path + "training.log", sep=",", engine="python")
    plot_loss_f1_score(log_data, model_path)
