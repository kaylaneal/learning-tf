# IMPORTS
import tensorflow as tf
import matplotlib.pyplot as plt

# DEFINE A TRAINING MONITOR
def get_training_monitor(test_dataset, output_path, stn_layername):
    (test_img, _) = next(iter(test_dataset))        # dataset has generator like properites, use next(iter())

    class TrainMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = None):
            model = tf.keras.models.Model(self.model.input, self.model.get_layer(stn_layername).output)
            test_pred = model(test_img)

            # plot images
            _, axes = plt.subplots(nrows = 5, ncols = 2, figsize = (5, 10))
            for ax, im, t_im, in zip(axes, test_img[:5], test_pred[:5]):
                ax[0].imshow(im[..., 0], cmap = 'gray')
                ax[0].set_title(epoch)
                ax[0].axis('off')

                ax[1].imshow(t_im[..., 0], cmap = 'gray')
                ax[1].set_title(epoch)
                ax[1].axis('off')
            
            plt.savefig(f'{output_path}/{epoch:03d}')
            plt.close()
    
    monitor = TrainMonitor()
    return monitor