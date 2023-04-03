# IMPORTS
import tensorflow as tf
import keras.layers as layers

# BUILD TRAINING MODEL
def get_training_model(batch_size, height, width, channel, stn_layer, num_classes, filter):
    # Input Layer --> STN Layer
    inputs = tf.keras.Input((height, width, channel), batch_size = batch_size)

    x = layers.Lambda(lambda image: tf.cast(image, 'float32')/255.0)(inputs)         # scales image to 0 and 1

    x = stn_layer(x)

    # Convolutional Block
    x = layers.Conv2D(filter // 4, 3, activation = 'relu',
                      kernel_initializer = 'he_normal')(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(filter // 2, 3, activation = 'relu',
                      kernel_initializer = 'he_normal')(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(filter, 3, activation = 'relu',
                      kernel_initializer = 'he_normal')(x)
    x = layers.MaxPool2D()(x)

    # Global Pool to Flatten
    x = layers.GlobalAveragePooling2D()(x)

    # Dense Layers
    x = layers.Dense(filter, activation = 'relu',
                     kernel_initializer = 'he_normal')(x)
    x = layers.Dense(filter // 2, activation = 'relu',
                     kernel_initializer = 'he_normal')(x)
    
    # Dropout
    x = layers.Dropout(0.5)(x)

    # Softmax to Classify
    outputs = layers.Dense(num_classes, activation = 'softmax')(x)

    return tf.keras.Model(inputs, outputs)