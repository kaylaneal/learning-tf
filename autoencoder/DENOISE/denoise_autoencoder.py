# IMPORTS
import tensorflow as tf
import matplotlib.pyplot as plt

## LOAD DATASET
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(f'Training Shape: {x_train.shape} \nTesting Shape: {x_test.shape}')

## ADD RANDOM NOISE TO IMAGES
noise_factor = 0.2

x_train_noisy = x_train + noise_factor * tf.random.normal(shape = x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape = x_test.shape)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min = 0., clip_value_max = 1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min = 0., clip_value_max = 1.)

## SHOW NOISY IMAGES (test that noise was applied)
n = 10
noisy_fig = plt.figure(figsize = (20, 2))
for i in range(n):
    noisy_fig.add_subplot(1, n, i+1)
    plt.title('Noisy Original Image')
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    plt.axis('off')
noisy_fig.savefig('DENOISE/noisy-images')

## CONVOLUTIONAL AUTOENCODER
class Denoise(tf.keras.Model):
    def __init__(self):
        super(Denoise, self).__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape = (28, 28, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', padding = 'same', strides = 2),
            tf.keras.layers.Conv2D(8, (3, 3), activation = 'relu', padding = 'same', strides = 2)
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(8, kernel_size = 3, strides = 2, activation = 'relu', padding = 'same'),
            tf.keras.layers.Conv2DTranspose(16, kernel_size = 3, strides = 2, activation = 'relu', padding = 'same'),
            tf.keras.layers.Conv2D(1, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

autoencoder = Denoise()
autoencoder.compile(optimizer = 'adam', loss = tf.keras.losses.MeanSquaredError())

history = autoencoder.fit(x_train_noisy, x_train, epochs = 10, shuffle = True, validation_data = (x_test_noisy, x_test))

## VALIDATION CURVE
val_fig = plt.figure()
val_fig.add_subplot(1, 1, 1)
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('AutoEncoder Loss')
plt.legend()
plt.savefig('DENOISE/validation_curve')

## RESULTS FIGURE
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

img_fig = plt.figure(figsize = (20, 4))
for i in range(n):
    # original
    img_fig.add_subplot(2, n, i+1)
    plt.imshow(x_test_noisy[i])
    plt.title('Noisy Original Image')
    plt.gray()
    plt.axis('off')

    # reconstruction
    img_fig.add_subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i])
    plt.title('Reconstructed')
    plt.gray()
    plt.axis('off')
img_fig.tight_layout()
plt.savefig('DENOISE/noiseog-vs-recon-10ep')

