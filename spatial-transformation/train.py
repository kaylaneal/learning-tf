# IMPORTS
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.callbacks import EarlyStopping

## LOCAL IMPORTS
import config
from callback import get_training_monitor
from classifier_model import get_training_model
from stn import STN

## CHECK FOR/CREATE DIRECTORIES:
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)
if not os.path.exists(config.DATASET_PATH):
    os.makedirs(config.DATASET_PATH)

# Set Seed for Reproducability
tf.random.set_seed(42)

# LOAD DATASETS:
print('** LOADING DATASETS **')
train_dataset = tfds.load(name = config.DATASET_NAME, data_dir = config.DATASET_PATH, 
                          split = 'train', shuffle_files = True, as_supervised = True)
test_dataset = tfds.load(name = config.DATASET_NAME, data_dir = config.DATASET_PATH,
                         split = 'test', as_supervised = True)

# PREPROCESS:
print('** PREPROCESSING DATASETS **')
train_dataset = (train_dataset
                 .shuffle(config.BATCH_SIZE*100)
                 .batch(config.BATCH_SIZE, drop_remainder = True)
                 .prefetch(config.AUTO))
test_dataset = (test_dataset
                .batch(config.BATCH_SIZE, drop_remainder = True)
                .prefetch(config.AUTO))

# BUILD  MODEL:
print('** INIT STN LAYER **')
stn_layer = STN(name = config.STN_LAYER_NAME, filter = config.FILTERS)

print('** INIT CLASSIFIER **')
model = get_training_model(batch_size = config.BATCH_SIZE, height = config.IMG_H, width = config.IMG_W,
                           channel = config.CH, stn_layer = stn_layer, 
                           num_classes = config.CLASSES, filter = config.FILTERS)

print(f'** MODEL SUMMARY: ** \n{model.summary()}')

# GET/DEFINE CALLBACKS:
monitor = get_training_monitor(test_dataset = test_dataset, 
                               output_path = config.OUTPUT_PATH, stn_layername = config.STN_LAYER_NAME)
earlystop = EarlyStopping(patience = 5, restore_best_weights = True)

# COMPILE AND TRAIN:
print('** COMPILING MODEL **')
model.compile(loss = config.LOSS_FN, optimizer = config.OPT, metrics = ['accuracy'])

print('** TRAINING MODEL **')
model.fit(train_dataset, epochs = config.EPOCHS, 
          callbacks = [monitor, earlystop], validation_data = test_dataset)
