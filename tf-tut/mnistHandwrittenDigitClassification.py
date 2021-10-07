# following a tutorial from https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/

from matplotlib import pyplot
from numpy import asarray
from numpy import unique
from numpy import argmax
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

# load dataset -- uses one of the imports! 
(xTrain, yTrain), (xTest, yTest) = load_data()

# summarize loaded dataset:
print('Train: X=%s, Y=%s' % (xTrain.shape, yTrain.shape))  # prints "Train: X=(60000, 28, 28), Y=(6000,)
print('Test: X=%s, Y=%s' % (xTest.shape, yTest.shape))  # prints "Test: X=(10000, 28, 28), Y=(1000,)

# plot first few images:
for i in range (25):
  pyplot.subplot(5, 5, i+1) # defines subplot
  pyplot.imshow(xTrain[i], cmap = pyplot.get_cmap('gray')) # plots raw pixel data
pyplot.show()

# reshape data to have a single channel:
xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1))
xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], xTest.shape[2], 1))

# determine the shape of the input images:
inShape = xTrain.shape[1:]

# determine the number of classes:
numClasses = len(unique(yTrain))

print(inShape, numClasses)  # prints: (28,28,1) 10

# normalize pixel values:
xTrain = xTrain.astype('float32') / 255.0
xTest = xTest.astype('float32') / 255.0

# define model:
model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', input_shape = inShape))
model.add(MaxPool2D((2,2))
model.add(Flatten())
model.add(Dense(100, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(Dropout(0.05))
model.add(Dense(numClasses, activation = 'softmax'))

# define loss and optimizer:
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# fit the model:
history = model.fit(xTrain, yTrain, epochs = 10, batch_size = 128, verbose = 0, validation_split = 0.2)

# evaluate the model:
loss, acc = model.evaluate(xTest, yTest, verbose = 0)
print ('Accuracy: %.3f' % acc)

# make a prediction:
image = xTrain[0]
yhat = model.predict(asarray([image]))
print('Predicted: class=%d' % argmax(yhat))

# you can use model.summary() to print a summary of each layer

# to plot learning curves:
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label = 'train')
pyplot.plot(history.history['val_loss'], label = 'val')
pyploy.legend()

# you can save this model to continue training by using model.save('whateverYouWantToTitleIt.h5')
