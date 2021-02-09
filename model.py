# Training data sets
# Final sets will come in three categories: center, recentre, special
import numpy as np
import csv
import cv2
import sklearn
import random
from scipy import ndimage
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Convolution2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

path_separator = '\\'
# Set our batch size
batch_size = 512
train_2_test_ratio = 5

def get_samples(training_folder):
    '''
    Parses the csv from the supplied training folder.
    '''
    samples = []
    with open(training_folder + '/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for sample in reader:
            samples.append(sample)
    return samples

# Samples are divided into three categories: centre, recentre and specials
# 'Centre' is data obtained by driving the car as closely in the middle of the track
# 'Recentre' is data obtained by driving from edges back to the middle of the road
# 'Specials' are used to augment correct behavior in special situation, ie when road ledges are partially missing...

def samples_selector(batch_size):
    '''
    Create even amount of data from every of the three groups.
    '''
    batch_size, batch_rem = divmod(batch_size, 3)
    centre_sampled = random.sample(centre_samples, batch_size)
    recentre_sampled = random.sample(recentre_samples, batch_size)
    special_sampled = random.sample(special_samples, batch_size + batch_rem)

    return centre_sampled + recentre_sampled + special_sampled

def samples_selector_valid(batch_size):
    '''
    Create even amount of data from every of the three groups.
    '''
    batch_size, batch_rem = divmod(batch_size, 3)
    centre_sampled = random.sample(centre_samples_valid, batch_size)
    recentre_sampled = random.sample(recentre_samples_valid, batch_size)
    special_sampled = random.sample(special_samples_valid, batch_size + batch_rem)

    return centre_sampled + recentre_sampled + special_sampled

def read_image(image_path):
    '''
    Read and grayscale the images.
    '''
    image = cv2.imread(image_path)
    # Grayscale the images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis = 2)

    return image

def generator(sample_selector, batch_size = 512):
    '''
    Generator prepares input data for the training of the NN. It also augments data with flipped inputs.
    '''

    batch_size = batch_size // 2 # Since we duplicate by fliping

    # Predefine the arrays, to prevent multiple re-instantiation
    X_train = np.empty((2 * batch_size, 160, 320, 1), dtype = float)
    y_train = np.empty((2 * batch_size), dtype = float)

    while 1: # Loop forever so the generator never terminates
        batch_samples = sample_selector(batch_size)

        counter = 0
        for batch_sample in batch_samples:
            # Get centre image only
            filename = batch_sample[0].split(path_separator)[-3:]
            current_path = path_separator.join(filename)
            center_image = read_image(current_path)
            center_angle = float(batch_sample[3])
            X_train[counter] = center_image
            y_train[counter] = center_angle
            counter = counter + 1

        # Augment with flipped data
        for i in range(batch_size):
            X_train[batch_size + i] = np.flip(X_train[i], 1)
            y_train[batch_size + i] = y_train[i] * -1.0

        yield tuple(shuffle(X_train, y_train))

def model_definition():
    # NVidia based model
    model = Sequential()

    input_shape = (160, 320, 1) # Inputs are grayscaled images
    cropping_shape = ((50,22), (0,0))
    droput_rate = 0.3

    model.add(Cropping2D(cropping = cropping_shape, input_shape = input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # First layer, Convolutional 3x3, Output = 44 x 160 x 3
    model.add(Convolution2D(24, 5, activation='relu', padding='same'))
    model.add(MaxPooling2D())

    # Second layer, Convolutional f36k5
    model.add(Convolution2D(36, 5, activation='relu'))
    model.add(MaxPooling2D())

    # Second layer, Convolutional f48k5
    model.add(Convolution2D(48, 5, activation='relu'))
    model.add(MaxPooling2D())

    # Second layer, Convolutional f64k3
    model.add(Convolution2D(64, 3, activation='relu'))

    # Second layer, Convolutional f64k3
    model.add(Convolution2D(64, 3, activation='relu'))

    # Flatten
    model.add(Flatten())

    # Fully connected
    model.add(Dense(100))

    # Dropout
    model.add(Dropout(droput_rate))

    # Fully connected
    model.add(Dense(50))

    # Fully connected
    model.add(Dense(1))

    # Summarize the model (https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/)
    # print(model.summary())

    model.compile(loss='mse', optimizer = 'adam')

    return model

# Create the model
model = model_definition()

# Create training samples (No actual data is provided of course)
centre_samples, centre_samples_valid = train_test_split(get_samples('training_data') + get_samples('training_data_2'), test_size = 1.0 / train_2_test_ratio)
recentre_samples, recentre_samples_valid = train_test_split(get_samples('training_data_recentre') + get_samples('training_data_recentre2'), test_size = 1.0 / train_2_test_ratio)
special_samples, special_samples_valid = train_test_split(get_samples('training_data_special') + get_samples('training_data_special2'), test_size = 1.0 / train_2_test_ratio)

training_samples_size = len(centre_samples) + len(recentre_samples) + len(special_samples)
validation_samples_size = training_samples_size // train_2_test_ratio

# compile and train the model using the generator function
train_generator = generator(samples_selector, batch_size = batch_size)
validation_generator = generator(samples_selector_valid, batch_size = batch_size // train_2_test_ratio)

# Training
model.fit_generator(train_generator,
            steps_per_epoch = np.ceil(training_samples_size / batch_size),
            validation_data = validation_generator,
            validation_steps = np.ceil(validation_samples_size / batch_size),
            epochs=5, verbose=1)
