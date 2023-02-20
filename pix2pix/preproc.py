import tensorflow as tf

AUTO = tf.data.AUTOTUNE

# function to load images:
def load_image(imageFile):
    image = tf.io.read_file(imageFile)
    image = tf.io.decode_jpeg(image, channels = 3)

    # data from this dataset has combined images;
    # split combined image:
    width = tf.shape(image)[1]
    mid = width // 2
    inputMask = image[:, mid:, :]
    realImage = image[:, :mid, :]

    # convert image to tensors and pixels to range [-1, 1]
    inputMask = tf.cast(inputMask, tf.float32)/127.5 -1
    realImage = tf.cast(realImage, tf.float32)/127.5 -1

    return (inputMask, realImage)

# function to create noise in generator dataset:
def rand_jitter(inputMask, realImage, height, width):
    # upscale:
    inputMask = tf.image.resize(inputMask, [height, width], 
                        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    realImage = tf.image.resize(realImage, [height, width], 
                        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return (inputMask, realImage)

class ReadTrainExample(object):

    def __init__(self, imageHeight, imageWidth):
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth

    def __call__(self, imageFile):
        # read in images
        inputMask, realImage = load_image(imageFile)

        # data augmentation:
        (inputMask, realImage) = rand_jitter(inputMask, realImage, self.imageHeight + 30, self.imageWidth + 30)

        # reshape:
        inputMask = tf.image.resize(inputMask, [self.imageHeight, self.imageWidth])
        realImage = tf.image.resize(realImage, [self.imageHeight, self.imageWidth])

        return (inputMask, realImage)

class ReadTestExample(object):

    def __init__(self, imageHeight, imageWidth):
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth

    def __call__(self, imageFile):
        # read in images
        inputMask, realImage = load_image(imageFile)

        # reshape:
        inputMask = tf.image.resize(inputMask, [self.imageHeight, self.imageWidth])
        realImage = tf.image.resize(realImage, [self.imageHeight, self.imageWidth])

        return (inputMask, realImage)

def load_dataset(path, batchSize, height, width, train = False):
    # check if training or testing:
    if train:
        # read training examples:
        dataset = tf.data.Dataset.list_files(str(path/"train/*.jpg"))
        dataset = dataset.map(ReadTrainExample(height, width), num_parallel_calls = AUTO)

    else:
        # read test examples:
        dataset = tf.data.Dataset.list_files(str(path/"val/*.jpg"))
        dataset = dataset.map(ReadTestExample(height, width), num_parallel_calls = AUTO)
    
    # Shuffle, Batch, Repeat, Prefetch:
    dataset = (dataset
                .shuffle(batchSize * 2)
                .batch(batchSize)
                .repeat()
                .prefetch(AUTO)
            )
    
    return dataset