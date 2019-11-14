# The full CNN code!
####################
import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

width = 1920
height = 1080
channels = 3
scale_factor = 0.2
scaled_width = int(width * scale_factor)
scaled_height = int(height * scale_factor)

# TODO: Subtract 0.5 from each pixel's intensity values
training_datagen = ImageDataGenerator(rescale = 1./255)

training_generator = training_datagen.flow_from_directory(
        'data/training/',
        target_size=(scaled_height, scaled_width),
        batch_size=32,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        'data/validation',
        target_size=(scaled_height, scaled_width),
        batch_size=32,
        class_mode='binary')

num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(scaled_height, scaled_width, channels)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(1, activation='sigmoid'),
])

# Compile the model.
model.compile(
  optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['accuracy']
)

model.fit_generator(
        training_generator,
        epochs=4,
        validation_data=validation_generator)

model.save_weights('train_detection_cnn.h5')
