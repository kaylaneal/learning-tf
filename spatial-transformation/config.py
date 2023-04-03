# IMPORTS
import tensorflow as tf

# GLOBALS
AUTO = tf.data.AUTOTUNE

# HEIGHT, WIDTH, CHANNEL
IMG_H = 28
IMG_W = 28
CH = 1

# PATHS
DATASET_PATH = 'dataset'
DATASET_NAME = 'emnist'
OUTPUT_PATH = 'outputs'

# HYPERPARAM
BATCH_SIZE = 1024

EPOCHS = 100
FILTERS = 256

LOSS_FN = 'sparse_categorical_crossentropy'
OPT = 'adam'

CLASSES = 62

STN_LAYER_NAME = 'stn'