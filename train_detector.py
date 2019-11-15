from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import time
from datetime import datetime

width = 1920
height = 1080
channels = 3
scale_factor = 0.2
scaled_width = int(width * scale_factor)
scaled_height = int(height * scale_factor)
num_filters = 16
filter_size = 3
pool_size = 2

model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(scaled_height, scaled_width, channels), activation='relu'),
  Conv2D(num_filters, filter_size, activation='relu'),
  MaxPooling2D(pool_size=pool_size),
  Dropout(0.5),
  Flatten(),
  Dense(1, activation='sigmoid'),
])
model.load_weights('train_detection_cnn.h5')

def detect_train(frame):
    prediction = (model.predict(frame)).flatten()[0]
    if prediction != 1.0:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S');
        print('{}: Prediction: {}'.format(current_time, prediction))

while (True):
    response = requests.get('http://10.10.1.181/axis-cgi/jpg/image.cgi?resolution={}x{}'.format(width, height))
    image = Image.open(BytesIO(response.content))
    image_resized = image.resize((scaled_width, scaled_height))
    frame = np.array(image_resized)
    frame_expanded = np.expand_dims(frame, axis=0)
    detect_train(frame_expanded)

    time.sleep(2)