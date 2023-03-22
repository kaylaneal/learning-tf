# AUTOENCODER TO DETECT ANOMALIES
# IMPORTS
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

'''
Electrocardiogram Dataset (ECG5000)
labeled as 0 (abnormal) or 1 (normal)
Autoencoder trained to minimize reconstruction error 
    -- abnormal would have high error, 
       categorize as such if error surpasses threshold
'''

## LOAD DATASET
df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header = None)
raw_data = df.values

print(f'ECG Dataframe: \n{df.head()}')

## UNSPERVISED -- remove labels
labels = raw_data[:, -1]

data = raw_data[:, 0:-1]                # data points

train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size = 0.2, random_state = 21)

## NORMALIZE DATA
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

## TRAIN **ONLY** ON NORMAL -- SEPARATE
train_label = train_label.astype(bool)
test_label = test_label.astype(bool)

normal_train_data = train_data[train_label]
normal_test_data = test_data[test_label]

abnom_train_data = train_data[~train_label]
abnom_test_data = test_data[~test_label]

## PLOT EXAMPLES OF NORMAL AND ABNORMAL
comp = plt.figure()

comp.add_subplot(2, 2, 1)
plt.plot(np.arange(140), normal_train_data[0])
plt.title('Normal ECG')
comp.add_subplot(2, 2, 2)
plt.plot(np.arange(140), normal_train_data[1])
plt.title('Normal ECG')

comp.add_subplot(2, 2, 3)
plt.plot(np.arange(140), abnom_train_data[0])
plt.title('Abnormal ECG')
comp.add_subplot(2, 2, 4)
plt.plot(np.arange(140), abnom_train_data[1])
plt.title('Abnormal ECG')

comp.tight_layout()
comp.savefig('DENOISE/normal-v-abnormal')

## AUTOENCODER
class AnomalyDetector(tf.keras.Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation = 'relu'),
            tf.keras.layers.Dense(16, activation = 'relu'),
            tf.keras.layers.Dense(8, activation = 'relu')
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation = 'relu'),
            tf.keras.layers.Dense(32, activation = 'relu'),
            tf.keras.layers.Dense(140, activation = 'sigmoid')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

autoencoder = AnomalyDetector()
autoencoder.compile(optimizer = 'adam', loss = 'mse')

history = autoencoder.fit(normal_train_data, normal_train_data,
                          epochs = 20, batch_size = 512,
                          validation_data = (test_data, test_data), shuffle = True)

## VALIDATION CURVE
val_fig = plt.figure()
val_fig.add_subplot(1, 1, 1)
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('AutoEncoder Loss')
plt.legend()
plt.savefig('ANOMALY_DETECT/validation_curve')

## VISUALIZE AUTOENCODER 
encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

ab_encoded_data = autoencoder.encoder(abnom_test_data).numpy()
ab_decoded_data = autoencoder.decoder(ab_encoded_data).numpy()

train_error_plt = plt.figure()

train_error_plt.add_subplot(1, 2, 1)
plt.plot(normal_test_data[0])
plt.plot(decoded_data[0])
plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color = 'lightgreen')
plt.legend(labels = ['Input', 'Reconstruction', 'Error'])
plt.title('Normal ECG + error')

train_error_plt.add_subplot(1, 2, 2)
plt.plot(abnom_test_data[0])
plt.plot(ab_decoded_data[0])
plt.fill_between(np.arange(140), ab_decoded_data[0], abnom_test_data[0], color = 'lightgreen')
plt.legend(labels = ['Input', 'Reconstruction', 'Error'])
plt.title('Abnormal ECG + error')

train_error_plt.savefig('ANOMALY_DETECT/error_plot')

## DETECT ANOMALIES
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
threshold = np.mean(train_loss) + np.std(train_loss)

ab_recon = autoencoder.predict(abnom_test_data)
test_loss = tf.keras.losses.mae(ab_recon, abnom_test_data)

loss_fig = plt.figure()

loss_fig.add_subplot(1, 2, 1)
plt.hist(train_loss[None, :], bins = 50)
plt.xlabel('Train Loss')
plt.ylabel('Num. Examples')
plt.axvline(threshold, color = 'purple')
plt.title('Normal Train Loss')

loss_fig.add_subplot(1, 2, 2)
plt.hist(test_loss[None, :], bins = 50)
plt.xlabel('Test Loss')
plt.ylabel('Num. Examples')
plt.axvline(threshold, color = 'purple')
plt.title('Normal Train Loss')

loss_fig.tight_layout()
loss_fig.savefig('ANOMALY_DETECT/losses_fig')

def predict(model, data, threshold):
    reconst = model(data)
    loss = tf.keras.losses.mae(reconst, data)

    return tf.math.less(loss, threshold)

def print_stats(preds, labels):
    print(f'Accuracy = {accuracy_score(labels, preds):.3f}')
    print(f'Precision = {precision_score(labels, preds):.3f}')
    print(f'Recall = {recall_score(labels, preds):.3f}')

predictions = predict(autoencoder, test_data, threshold)
print_stats(predictions, test_label)
print(f'Threshold = {threshold:.3f}')