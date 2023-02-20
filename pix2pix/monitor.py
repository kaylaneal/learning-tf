from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import tensorflow as tf

def get_train_monitor(testD, imagePath, batchSize, epochInterval):
    # input mask and real image
    (tInputMask, tRealImage) = next(iter(testD))

    class TrainMonitor(Callback):
        def __init__(self, epochInterval = None):
            self.epochInterval = epochInterval

        def on_epoch_end(self, epoch, logs = None):
            if self.epochInterval and epoch % self.epochInterval == 0:
                tpixGenPred = self.model.generator.predict(tInputMask)

                (fig, axes) = subplots(nrows = batchSize, ncols = 3, figsize = (50, 50))

                # Plot Predicted Images:
                for (ax, inp, pred, tgt) in zip(axes, tInputMask, tpixGenPred, tRealImage):
                    ax[0].imshow(array_to_img(inp))
                    ax[0].set_title("Input Image")

                    ax[1].imshow(array_to_img(pred))
                    ax[1].set_title("Pix2Pix Prediction")

                    ax[2].imshow(array_to_img(tgt))
                    ax[2].set_title("Target Label/Ground Truth")

                plt.savefig(f"{imagePath}/{epoch:03d}.png")

    trainMonitor = TrainMonitor(epochInterval = epochInterval)
    return trainMonitor