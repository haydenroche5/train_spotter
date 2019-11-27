from PIL import Image
from io import BytesIO
import numpy as np
import requests
import time
import logging
import argparse
from tensorflow.keras.models import load_model
from requests.exceptions import ConnectionError


def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('predictions.log', mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                  datefmt='%m-%d-%Y %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    width = 1920
    height = 1080
    threshold = 0.96
    logger.info('THRESHOLD: {}'.format(threshold))

    model = load_model(args.model_dir)
    input_img_height, input_img_width = model.layers[0].input_shape[1:3]

    while (True):
        try:
            response = requests.get(
                'http://{}/axis-cgi/jpg/image.cgi?resolution={}x{}'.format(
                    args.camera_ip, width, height),
                timeout=5)
            image = Image.open(BytesIO(response.content))
            image_resized = image.resize((input_img_width, input_img_height))
            frame = np.array(image_resized)
            frame_scaled = frame / 255.0
            frame_scaled_expanded = np.expand_dims(frame_scaled, axis=0)
            prediction_value = model.predict(
                frame_scaled_expanded).flatten()[0]
            if prediction_value > threshold:
                logger.info('Prediction value: {}'.format(prediction_value))
                logger.info('TRAIN!')
            time.sleep(3)
        except ConnectionError:
            raise Exception(
                'Timed out trying to get an image from the webcam.')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Retrain the train detection model.')
    arg_parser.add_argument('--model-dir',
                            dest='model_dir',
                            required=True,
                            help='Directory containing model.')
    arg_parser.add_argument('--camera-ip',
                            dest='camera_ip',
                            required=True,
                            help='IP address of the webcam.')
    main(arg_parser.parse_args())