from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model
from tensorflow.keras import Input

class Pix2Pix(object):
    def __init__(self, imageHeight, imageWidth):
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth

    def generator(self):
        inputs = Input([self.imageHeight, self.imageWidth, 3])

        # Down Layer 1 => Final Layer 1
        d1 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(inputs)
        d1 = Dropout(0.1)(d1)
        f1 = MaxPool2D((2, 2))(d1)

        # Down Layer 2 => Final Layer 2
        d2 = Conv2D(64, (3, 3), activation = "relu", padding = "same")(f1)
        f2 = MaxPool2D((2,2))(d2)

        # Down Layer 3 => Final Layer 3
        d3 = Conv2D(64, (3, 3), activation = "relu", padding = "same")(f2)
        f3 = MaxPool2D((2,2))(d3)

        # Down Layer 4 => Final Layer 4
        d4 = Conv2D(64, (3, 3), activation = "relu", padding = "same")(f3)
        f4 = MaxPool2D((2,2))(d4)

        # U of UNet
        b5 = Conv2D(96, (3, 3), activation = "relu", padding = "same")(f4)
        b5 = Dropout(0.3)(b5)
        b5 = Conv2D(256, (3, 3), activation = "relu", padding = "same")(b5)

        # Upsample Layer 6
        u6 = Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = "same")(b5)
        u6 = concatenate([u6, d4])
        u6 = Conv2D(128, (3, 3), activation = "relu", padding = "same")(u6)

        # Upsample Layer 7
        u7 = Conv2DTranspose(96, (2, 2), strides = (2, 2), padding = "same")(u6)
        u7 = concatenate([u7, d3])
        u7 = Conv2D(128, (3, 3), activation = "relu", padding = "same")(u7)

        # Upsample Layer 8
        u8 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = "same")(u7)
        u8 = concatenate([u8, d2])
        u8 = Conv2D(128, (3, 3), activation = "relu", padding = "same")(u8)

        # Upsample Layer 9
        u9 = Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = "same")(u8)
        u9 = concatenate([u9, d1])
        u9 = Dropout(0.1)(u9)
        u9 = Conv2D(128, (3, 3), activation = "relu", padding = "same")(u9)

        # Final Conv2D Layer
        outputLayer = Conv2D(3, (1, 1), activation = "tanh")(u9)

        generator = Model(inputs, outputLayer)
        return generator


    def discriminator(self):
        # initialize input layer
        inputMask = Input(shape = [self.imageHeight, self.imageWidth, 3], name = "input_image")
        targetImage = Input(shape = [self.imageHeight, self.imageWidth, 3], name = "target_image")

        # concatenate inputs
        x = concatenate([inputMask, targetImage])

        # add Conv2D Layers
        x = Conv2D(64, 4, strides = 2, padding = "same")(x)
        x = LeakyReLU()(x)
        x = Conv2D(128, 4, strides = 2, padding = "same")(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 4, strides = 2, padding = "same")(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, 4, strides = 1, padding = "same")(x)

        # add Batch Normalization, Leaky ReLU, zeropad
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Final Layer
        last = Conv2D(1, 3, strides = 1)(x)

        discriminator = Model(inputs = [inputMask, targetImage], outputs = last)
        return discriminator
