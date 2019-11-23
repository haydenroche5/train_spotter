from PIL import Image
from io import BytesIO
import numpy as np
import requests
import time
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from requests.exceptions import ConnectionError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('predictions.log', mode = 'w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt = '%m-%d-%Y %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

width = 1920
height = 1080

model = load_model('train_detection_cnn')
input_img_height, input_img_width = model.layers[0].input_shape[1:3]

def detect_train(frame):
    prediction_value = model.predict(frame).flatten()[0]
    threshold = 0.98
    if prediction_value > threshold:
        logger.info('Prediction value: {}'.format(prediction_value))
        logger.info('Prediction: train')
    # else:
    #     logger.info('Prediction: no_train')

while (True):
    try:
        response = requests.get('http://10.10.1.181/axis-cgi/jpg/image.cgi?resolution={}x{}'.format(width, height), timeout = 5)
        image = Image.open(BytesIO(response.content))
        image_resized = image.resize((input_img_width, input_img_height))
        frame = np.array(image_resized)
        frame_scaled = frame / 255.0
        frame_scaled_expanded = np.expand_dims(frame_scaled, axis = 0)
        detect_train(frame_scaled_expanded)
        time.sleep(1)
    except ConnectionError:
        print('Timed out trying to get an image from the webcam.')