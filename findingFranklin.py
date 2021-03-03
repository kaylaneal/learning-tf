import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
# import all the requires packages 
# we will import more as we go along, but it will be with what we use it for for clarity

model = VGG19()
# this creates the model

from tensorflow.keras.preprocessing.image import load_img
# the machinery to load the file

testingImage = load_img('frank.jpeg', target_size = (224,224))
# loads and resizes the image to required pixel size
# frank.jpeg is a specific example, you would load whatever the name of the image is

from tensorflow.keras.preprocessing.image import img_to_array
# machinery to convert the image pixels to a numpy array

testingImage = img_to_array(testingImage)

testingImage = testingImage.reshape((1, testingImage.shape[0], testingImage.shape[1], testingImage.shape[2]))
# reshapes image to be 4 dimensions

from tensorflow.keras.applications.vgg19 import preprocess_input
# function will prepare new imput for the network

testingImage = preprocess_input(testingImage)

testPrediction = model.predict(testingImage)
# predict the probability across all output classes

from tensorflow.keras.applications.vgg19 import decode_predictions
# function can return a list of classes and their probablities

testingLabel = decode_predictions(testPrediction)
# coverts the probailities to class labels

testingLabel = testingLabel [0][0]
# retrieve the most likely result (highest probability)

print('%s (%.2f%%)' % (testingLabel[2]*100))
# prints the classification
