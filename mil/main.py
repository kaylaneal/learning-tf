# KERAS MIL ATTENTION MODEL IMPLEMENTATION -- https://keras.io/examples/vision/attention_mil_classification/
from train import train
from model import create_model, visualize
from bags import BAG_COUNT, BAG_SIZE, VAL_BAG_COUNT, ENSEMBLE_AVG_COUNT, POSITIVE_CLASS, make_bags
from eval import make_predictions
from tqdm import tqdm
from tensorflow import keras

# Load Data
print("****** LOADING DATA ******")
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

train_data, train_labels = make_bags(x_train, y_train, POSITIVE_CLASS, BAG_COUNT, BAG_SIZE)
val_data, val_labels = make_bags(x_val, y_val, POSITIVE_CLASS, VAL_BAG_COUNT, BAG_SIZE)

# Build Model
print("****** BUILDING MODEL ******")
instance_shape = train_data[0][0].shape
models = [create_model(instance_shape) for _ in range(ENSEMBLE_AVG_COUNT)]

# models[0].summary()           # prints the first model summary

# Train
print("****** TRAINING ******")
trained_models = [
    train(train_data, train_labels, val_data, val_labels, model)
    for model in tqdm(models)
]

# Evaluate
print("****** EVALUATE PREDICTIONS ******")
class_predictions, attention_params = make_predictions(val_data, val_labels, trained_models)
'''
# Visualize
print("****** PLOTTING ******")
visualize(
    val_data, val_labels, "positive",
    predictions = class_predictions, attention_weights = attention_params
)

visualize(
    val_data, val_labels, "negative",
    predictions = class_predictions, attention_weights = attention_params
)'''