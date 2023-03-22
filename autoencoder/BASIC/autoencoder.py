# TENSORFLOW TUTORIALS: INTRODUCTION TO AUTOENCODERS -- https://www.tensorflow.org/tutorials/generative/autoencoder
# IMPORTS:
import tensorflow as tf
import matplotlib.pyplot as plt

## LOAD DATASET
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(f'Training Shape: {x_train.shape} \nTesting Shape: {x_test.shape}')

## BASIC AUTOENCODER ##
latent_dim = 64

class AutoEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim, activation = 'relu')
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(784, activation = 'sigmoid'),
            tf.keras.layers.Reshape((28, 28))                        # original input size
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

autoencoder = AutoEncoder(latent_dim)
autoencoder.compile(optimizer = 'adam', loss = tf.keras.losses.MeanSquaredError())

history = autoencoder.fit(x_train, x_train, epochs = 10, shuffle = True, validation_data = (x_test, x_test))              # autoencoders set x and y to be equal; idea is there is not label to use for y

val_fig = plt.figure()
val_fig.add_subplot(1, 1, 1)
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('AutoEncoder Loss')
plt.legend()
plt.savefig('BASIC/validation_curve')

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
img_fig = plt.figure(figsize = (20, 4))
for i in range(n):
    # original
    img_fig.add_subplot(2, n, i+1)
    plt.imshow(x_test[i])
    plt.title('Original')
    plt.gray()
    plt.axis('off')

    # reconstruction
    img_fig.add_subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i])
    plt.title('Reconstructed')
    plt.gray()
    plt.axis('off')
img_fig.tight_layout()
plt.savefig('BASIC/og-vs-recon-10ep')
