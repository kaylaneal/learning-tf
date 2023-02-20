import tensorflow as tf

import config
from pixGAN import Pix2Pix
from preproc import load_dataset
from pixPipe import PixTraining
from monitor import get_train_monitor

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.utils import get_file

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

# Build Training Set
print("[INFO] Building Train Dataset ... ")
traind = load_dataset(path = path, train = True, batchSize = config.TRAIN_BATCH_SIZE, 
                        height = config.IMAGE_HEIGHT, width = config.IMAGE_WIDTH)

# Build Testing Set
testd = load_dataset(path = path, train = False, batchSize = config.INFER_BATCH_SIZE, 
                        height = config.IMAGE_HEIGHT, width = config.IMAGE_WIDTH)

# Initialize the Generator and Discriminator Network
print("[INFO] Initializing the Generator and Discriminator ... ")
p2pObj = Pix2Pix(imageHeight = config.IMAGE_HEIGHT, imageWidth = config.IMAGE_WIDTH)
generator = p2pObj.generator()
discriminator = p2pObj.discriminator()

# Build and Compile Training Model
model = PixTraining( generator = generator, discriminator = discriminator)
model.compile  (gOpt = Adam(learning_rate = config.LEARNING_RATE),
                dOpt = Adam(learning_rate = config.LEARNING_RATE),
                bceLoss = BinaryCrossentropy(from_logits = True),
                maeLoss = MeanAbsoluteError()
                )

if not os.path.exists(config.BASE_OUTPUT_PATH):
    os.makedirs(config.BASE_OUTPUT_PATH)
if not os.path.exists(config.BASE_IMAGES_PATH):
    os.makedirs(config.BASE_IMAGES_PATH)

# Train Model
print("[INFO] Training ... ")
callbacks = [get_train_monitor(testd, epochInterval = 10, 
                imagePath = config.BASE_IMAGES_PATH, batchSize = config.INFER_BATCH_SIZE)]
model.fit(traind, epochs = config.EPOCHS, callbacks = callbacks, steps_per_epoch = config.STEPS_PER_EPOCH)

# Set path for generator
genPath = config.GENERATOR_MODEL

print(f"[INFO] Saving Pix2Pix Generator to {genPath} ... ")
model.generator.save(genPath)