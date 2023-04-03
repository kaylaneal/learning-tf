# INPUTS
import tensorflow as tf
import keras.layers as layers

'''
GOAL:
- retrieve transformation parameters from input image
- map input feature map from output feature map
- apply bilinear interpolation to estimate output feature map values
'''

# Retrieve Pixel Value
def get_pixel_value(B, H, W, feature_map, x, y):
    # batch indicies:
    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1))

    # indicies matrix to sample feature map:
    b = tf.tile(batch_idx, (1, H, W))
    indicies = tf.stack([b, y, x], 3)

    # gather and return feature map values at indicies:
    pixel_value = tf.gather_nd(feature_map, indicies)

    return pixel_value

# Affine Grid Generator: map output coordinate to input coordinate
def affine_grid_generator(B, W, H, theta):
    # 2D Grid
    x = tf.linspace(-1.0, 1.0, H)
    y = tf.linspace(-1.0, 1.0, W)
    (xT, yT) = tf.meshgrid(x, y)

    # Flatten Grid
    xT_flat = tf.reshape(xT, [-1])
    yT_flat = tf.reshape(yT, [-1])

    # Homogenous form
    ones = tf.ones_like(xT_flat)
    sampling_grid = tf.stack([xT_flat, yT_flat, ones])

    # MatMul needs float32:
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # Transform Sampling Grid with Affine Parameters (Theta)
    batch_grid = tf.matmul(theta, sampling_grid)
    batch_grid = tf.reshape(batch_grid, [B, 2, H, W])

    return batch_grid

# Interpolate to get final output
def bilinear_sampler(B, H, W, feature_map, x, y):
    # image bounds
    max_x = tf.cast(H - 1, 'int32')
    max_y = tf.cast(W - 1, 'int32')
    zeroes = tf.zeros([], dtype = 'int32')

    # rescale to feature dimensions
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))

    # 4 Nearest corner points for each x, y
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # Clip to stay in bounds
    x0 = tf.clip_by_value(x0, zeroes, max_x)
    x1 = tf.clip_by_value(x1, zeroes, max_x)
    y0 = tf.clip_by_value(y0, zeroes, max_y)
    y1 = tf.clip_by_value(y1, zeroes, max_y)

    # get values for corner coords:
    Ia = get_pixel_value(B, H, W, feature_map, x0, y0)
    Ib = get_pixel_value(B, H, W, feature_map, x0, y1)
    Ic = get_pixel_value(B, H, W, feature_map, x1, y0)
    Id = get_pixel_value(B, H, W, feature_map, x1, y1)

    # Calculate Deltas
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    wa = tf.expand_dims(wa, axis = 3)
    wb = tf.expand_dims(wb, axis = 3)
    wc = tf.expand_dims(wc, axis = 3)
    wd = tf.expand_dims(wd, axis = 3)

    # Compute Transformed Feature Map
    transformed_featuremap = tf.add_n(
        [wa * Ia, wb * Ib,
         wc * Ic, wd * Id]
    )

    return transformed_featuremap

## CREATE STN CLASS
class STN(layers.Layer):
    def __init__(self, name, filter):
        super().__init__(name = name)
        self.B = None
        self.H = None
        self.W = None
        self.C = None

        self.filter = filter

        self.output_bias = tf.keras.initializers.Constant(
            [ 1.0, 0.0, 0.0,
             0.0, 1.0, 0.0 ]
        )
    
    def build(self, input_shape):
        (self.B, self.H, self.W, self.C) = input_shape

        # Define Localization Network
        self.localization_net = tf.keras.Sequential([
            layers.Conv2D(filters =  self.filter // 4, kernel_size = 3,
                          input_shape = (self.H, self.W, self.C),
                          activation = 'relu', kernel_initializer = 'he_normal'),
            layers.MaxPool2D(),
            layers.Conv2D(filters = self.filter // 2, kernel_size = 3, 
                          activation = 'relu', kernel_initializer = 'he_normal'),
            layers.MaxPool2D(),
            layers.Conv2D(filters = self.filter // 2, kernel_size = 3, 
                          activation = 'relu', kernel_initializer = 'he_normal'),
            layers.MaxPool2D(),
            layers.GlobalAveragePooling2D()
        ])
        # output = feature map

        # Define Regressor Network
        self.regressor_net = tf.keras.Sequential([
            layers.Dense(units = self.filter, activation = 'relu',
                         kernel_initializer = 'he_normal'),
            layers.Dense(units = self.filter // 2, activation = 'relu',
                         kernel_initializer = 'he_normal'),
            layers.Dense(units = 3 * 2, kernel_initializer = 'zeros',
                         bias_initializer = self.output_bias),
            layers.Reshape(target_shape = (2, 3))
        ])
        # output = theta
    
    def call(self, x):
        # Get Localization Feature Map
        local_featmap = self.localization_net(x)

        # Get Regressed Parameters:
        theta = self.regressor_net(local_featmap)

        # Get Transformed Meshgrid + Coordinates
        grid = affine_grid_generator(self.B, self.H, self.W, theta)
        xS = grid[:, 0, :, :]
        yS = grid[:, 1, :, :]

        # Get Transformed Feature Map
        x = bilinear_sampler(self.B, self.H, self.W, x, xS, yS)
        return x
    
