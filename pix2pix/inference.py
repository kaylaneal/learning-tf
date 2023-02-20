import tensorflow as tf

import config
from preproc import load_dataset

from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from matplotlib.pyplot import subplots

import matplotlib.pyplot as plt
import pathlib
import os

tf.random.set_seed(42)

# Download Dataset
print("[INFO] Downloading Dataset ... ")
pathToZip = get_file(
    fname = f"{config.DATASET}.tar.gz",
    origin = config.DATASET_URL, extract = True
)
pathToZip = pathlib.Path(pathToZip)
path = pathToZip.parent/config.DATASET

# Build Test Dataset
print("[INFO] Building Test Dataset ... ")
testd = load_dataset(path = path, train = False, batchSize = config.INFER_BATCH_SIZE,
                        height = config.IMAGE_HEIGHT, width = config.IMAGE_WIDTH)

# First Batch of Testing Images
(inputMask, realImage) = next(iter(testd))

# Set Gen Path
genPath = config.GENERATOR_MODEL

# Load Trained Generator
print("[INFO] Loading Trained Pix2Pix Generator ... ")
p2pG = load_model(genPath, compile = False)

# Predict
print("[INFO] Making Predictions ... ")
p2pGpred = p2pG.predict(inputMask)

# Plot Predictions
print("[INFO] Saving Predictions ... ")
(fig, axes) = subplots(nrows = config.INFER_BATCH_SIZE, ncols = 3, figsize = (50, 50))

for (ax, inp, pred, tar) in zip(axes, inputMask, p2pGpred, realImage):
    ax[0].imshow(array_to_img(inp))
    ax[0].set_title("Input Image")

    ax[1].imshow(array_to_img(pred))
    ax[1].set_title("Pix2Pix Prediction")

    ax[2].imshow(array_to_img(tgt))
    ax[2].set_title("Target Label/Ground Truth")

if not os.path.exists(config.BASE_IMAGES_PATH):
    os.makedirs(config.BASE_IMAGES_PATH)

print("[INFO] Saving Predictions ... ")
fig.savefig(config.GRID_IMAGE_PATH)