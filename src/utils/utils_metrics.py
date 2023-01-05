""" Add utils function for F1 metric. """

from keras import backend as K


def recall_metrics(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metrics(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_metrics(y_true, y_pred):
    precision = precision_metrics(y_true, y_pred)
    recall = recall_metrics(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
