# INPUTS
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from model import compute_class_weights

def train(train_data, train_labels, val_data, val_labels, model):
    # Path to save weights to:
    weight_path = '/tmp/best_weights.h5'

    # Callbacks
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        weight_path, monitor = "val_loss", verbose = 0,
        mode = "min", save_best_only = True, save_weights_only = True
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor = "val_loss", patience = 10, mode = "min"
    )

    callbacks = [model_checkpoint, early_stopping]

    # Compile:
    model.compile(
        optimizer = "adam", loss = "sparse_categorical_crossentropy",
        metrics = ['accuracy']
    )

    # Fit:
    model.fit(
        train_data, train_labels,
        validation_data = (val_data, val_labels),
        epochs = 20, class_weight = compute_class_weights(train_labels),
        batch_size = 1, callbacks = callbacks, verbose = 0
    )

    # Load Best Weights and Return Model
    model.load_weights(weight_path)

    return model