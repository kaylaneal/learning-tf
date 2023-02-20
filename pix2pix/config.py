# Configuration Pipeline:
import os

DATASET = "cityscapes"
DATASET_URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{DATASET}.tar.gz"

TRAIN_BATCH_SIZE = 32
INFER_BATCH_SIZE = 8

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNELS = 3

LEARNING_RATE = 2e-4
EPOCHS = 150
STEPS_PER_EPOCH = 100

BASE_OUTPUT_PATH = "outputs"

GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH, "models", "generator")

BASE_IMAGES_PATH = os.path.join(BASE_OUTPUT_PATH, "images")
GRID_IMAGES_PATH = os.path.join(BASE_IMAGES_PATH, "grid.png")


# Tut. https://pyimagesearch.com/2022/07/27/image-translation-with-pix2pix/
