# https://www.tensorflow.org/tutorials/images/segmentation

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt

# DOWNLOAD DATASET
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info = True)

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32)/255.0
    input_mask -= 1
    return input_image, input_mask

def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128,128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128,128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode = "horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode = "horizontal", seed = seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels

# BUILD INPUT PIPELINE
train_batches = (train_images.
                take(BATCH_SIZE).
                cache().
                shuffle(BUFFER_SIZE).
                repeat().
                batch(BATCH_SIZE).
                map(Augment()).
                prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
                )

test_batches = test_images.batch(BATCH_SIZE)

# VISUALIZATION 
def display(display_list):
    plt.figure(figsize = (15,15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")

    plt.show()

# show sample input // true mask
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
#    display([sample_image, sample_mask])

# BUILDING UNET MODEL
base_model = tf.keras.applications.MobileNetV2(input_shape = [128, 128, 3], include_top = False)

# use activations of:
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# feature extraction
down_stack = tf.keras.Model(inputs = base_model.input, outputs = base_model_outputs)
down_stack.trainable = False

# decoder / upsampler
up_stack = [
    pix2pix.upsample(512, 3),   # 4x4 -> 8x8
    pix2pix.upsample(256, 3),   # 8x8 -> 16x16
    pix2pix.upsample(128, 3),   # 16x16 -> 32x32
    pix2pix.upsample(64, 3),    # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape = [128, 128, 3])

    # downsampling
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # upsampling & skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # last layer
    last = tf.keras.layers.Conv2DTranspose(
        filters = output_channels, kernel_size = 3,
        strides = 2, padding = 'same' ) # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs = inputs, outputs = x)

# Training!
OUTPUT_CLASSES = 3

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(
    optimizer = "adam", 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ["accuracy"])

# to view architecture:
# tf.keras.utils.plot_model(model, show_shapes = True)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis = -1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset = None, num = 1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])

    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        clear_output(wait = True)
        show_predictions()
        print("\nSample Prediction after epoch {}\n".format(epoch+1))

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(
    train_batches, epochs = EPOCHS,
    steps_per_epoch = STEPS_PER_EPOCH,
    validation_steps = VALIDATION_STEPS,
    validation_data = test_batches,
    callbacks = [DisplayCallback()]
)

# Training Curve:
loss = model_history.history['loss']
val_loss = model_history.history['loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label = "Training Loss")
plt.plot(model_history.epoch, val_loss, 'bo', label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.ylim([0,1])
plt.legend()
plt.show()


# PREDICTIONS
show_predictions(test_batches, 3)