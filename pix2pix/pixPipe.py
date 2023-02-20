from tensorflow.keras import Model
import tensorflow as tf

class PixTraining(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, gOpt, dOpt, bceLoss, maeLoss):
        super().compile()

        self.gOpt = gOpt
        self.dOpt = dOpt
        self.bceLoss = bceLoss
        self.maeLoss = maeLoss

    def train_step(self, inputs):
        (inputMask, realImages) = inputs

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            fakeImages = self.generator(inputMask, True)

            discRealOutput = self.discriminator([inputMask, realImages], True)
            discFakeOutput = self.discriminator([inputMask, fakeImages], True)

            # compute adversarial loss for generator
            misleadingLabels = tf.ones_like(discFakeOutput)
            ganLoss = self.bceLoss(misleadingLabels, discFakeOutput)

            # compute MAE between fake and real
            l1Loss = self.maeLoss(realImages, fakeImages)

            # compute total generator loss
            totalGenLoss = ganLoss + (10 * l1Loss)

            # discriminator loss
            realLabels = tf.ones_like(discRealOutput)
            realDiscLoss = self.bceLoss(realLabels, discRealOutput)
            fakeLabels = tf.zeros_like(discFakeOutput)
            generatedLoss = self.bceLoss(fakeLabels, discFakeOutput)

            totalDiscLoss = realDiscLoss + generatedLoss

        # calculate gradients
        generatorGradient = genTape.gradient(totalGenLoss, self.generator.trainable_variables)
        discriminatorGradient = discTape.gradient(totalDiscLoss, self.discriminator.trainable_variables)

        # apply gradients to optimize
        self.gOpt.apply_gradients(zip(generatorGradient, self.generator.trainable_variables))
        self.dOpt.apply_gradients(zip(discriminatorGradient, self.discriminator.trainable_variables))

        return {"dLoss": totalDiscLoss, "gLoss": totalGenLoss}