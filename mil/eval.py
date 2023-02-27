# IMPORTS:
import tensorflow as tf
from tensorflow import keras
import numpy as np
from bags import ENSEMBLE_AVG_COUNT

def make_predictions(data, labels, trained):
    # Statistics from each Trained Model:
    models_preds = []
    models_att_weights = []
    models_losses = []
    models_acc = []

    for model in trained:
        # Predict Output Classes:
        predictions = model.predict(data)
        models_preds.append(predictions)

        # Intermediate Model to get Attention Weights:
        inter_model = keras.Model(model.input, model.get_layer("alpha").output)
        inter_pred = inter_model.predict(data)

        attention_weights = np.squeeze(np.swapaxes(inter_pred, 1, 0))
        models_att_weights.append(attention_weights)

        # Loss and Accuracy:
        loss, accuracy = model.evaluate(data, labels, verbose = 0)
        models_acc.append(accuracy)
        models_losses.append(loss)

    print(f"Average Loss: {np.sum(models_losses, axis = 0) / ENSEMBLE_AVG_COUNT:.2f} ")
    print(f"Average Accuracy: {100 * np.sum(models_acc, axis = 0) / ENSEMBLE_AVG_COUNT:.3f}%")

    return (
        np.sum(models_losses, axis = 0) / ENSEMBLE_AVG_COUNT,
        np.sum(models_acc, axis = 0) / ENSEMBLE_AVG_COUNT
        ) 
