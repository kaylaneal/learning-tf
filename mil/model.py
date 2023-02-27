# INPUTS:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from bags import PLOT_SIZE, BAG_SIZE


# Create MIL Attention Layer:

class AttentionLayer(layers.Layer):
    def __init__(self, weight_param_dims, kernel_initializer = keras.initializers.glorot_uniform, kernel_regualizer = None, gated = False, **kwargs):
        super().__init__(**kwargs)

        self.weight_param_dims = weight_param_dims
        self.gated = gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regualizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_reg = self.kernel_regularizer
        self.w_reg = self.kernel_regularizer
        self.u_reg = self.kernel_regularizer

    def build(self, input_shape):

        input_dims = input_shape[0][1]

        self.v_weight_params = self.add_weight( shape = (input_dims, self.weight_param_dims),
                                                initializer = self.v_init, name = "v", 
                                                regularizer = self.v_reg, trainable = True )
        
        self.w_weight_params = self.add_weight( shape = (self.weight_param_dims, 1),
                                                initializer = self.w_init, name = "w", 
                                                regularizer = self.w_reg, trainable = True )

        if self.gated:
            self.u_weight_params = self.add_weight( shape = (input_dims, self.weight_param_dims),
                                                initializer = self.u_init, name = "u", 
                                                regularizer = self.u_reg, trainable = True )
        else:
            self.u_weight_params = None
        self.input_built = True

    def call(self, inputs):
        # Assign Variables
        instances = [self.compute_score(instance) for instance in inputs]

        # Softmax instances such that sum is 1
        alpha = tf.math.softmax(instances, axis = 0)

        return [alpha[i] for i in range(alpha.shape[0])]
    
    def compute_score(self, instance):
        orignial = instance

        instance = tf.math.tanh(
            tf.tensordot(instance, self.v_weight_params, axes = 1)
        )

        if self.gated:
            instance = instance * tf.math.sigmoid(
                tf.tensordot(orignial, self.u_weight_params, axes = 1)          # tanh(v*h_k^T)
            )
        
        return tf.tensordot(instance, self.w_weight_params, axes = 1)           # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))

# Plotting Function to Visualize Bags:

def visualize(data, labels, bag_class, predictions = None, attention_weights = None):
    
    labels = np.array(labels).reshape(-1)

    if bag_class == "positive":
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

        else:
            labels = np.where(labels == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

    elif bag_class == "negative":
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
        else:
            labels = np.where(labels == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

    else:
        print(f"There is no class {bag_class}")
        return

    print(f"The bag class label is {bag_class}")
    for i in range(PLOT_SIZE):
        figure = plt.figure(figsize=(8, 8))
        print(f"Bag number: {labels[i]}")
        for j in range(BAG_SIZE):
            image = bags[j][i]
            figure.add_subplot(1, BAG_SIZE, j + 1)
            plt.grid(False)
            if attention_weights is not None:
                plt.title(np.around(attention_weights[labels[i]][j], 2))
            plt.imshow(image)
        plt.show()

def create_model(instance_shape):

    # Extract Features from Input
    inputs, embeddings = [], []

    shared_dense1 = layers.Dense(128, activation = "relu")
    shared_dense2 = layers.Dense(64, activation = "relu")

    for _ in range(BAG_SIZE):
        inp = layers.Input(instance_shape)
        flt = layers.Flatten()(inp)
        d1 = shared_dense1(flt)
        d2 = shared_dense2(d1)
        inputs.append(inp)
        embeddings.append(d2)
    
    # Attention Layer
    alpha = AttentionLayer(
        weight_param_dims = 256,
        kernel_regualizer = keras.regularizers.l2(0.01),
        gated = True, name = "alpha"
    )(embeddings)

    # Multiply Attention Weights and Inputs
    mult = [
        layers.multiply(
            [alpha[i], embeddings[i]]
        ) for i in range(len(alpha))
    ]

    # Add Layers
    concat = layers.concatenate(mult, axis = 1)

    # Classify
    out = layers.Dense(2, activation = "softmax")(concat)

    return keras.Model(inputs, out)

def compute_class_weights(labels):
    # Counts
    npos = len(np.where(labels == 1)[0])
    nneg = len(np.where(labels == 0)[0])
    total = npos + nneg

    # Class Weight Dictionary
    return {
        0: (1 / nneg) * (total / 2),
        1: (1 / npos) * (total / 2)
    }