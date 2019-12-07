from PIL import Image
from io import BytesIO
import numpy as np
import requests
import time
import logging
import argparse
from tensorflow.keras.models import load_model
from requests.exceptions import ConnectionError
from datetime import datetime
import pickle
import os
import re


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
    logger.info('THRESHOLD: {}'.format(args.threshold))

    model = load_model(args.model_dir)
    input_img_height, input_img_width = model.layers[0].input_shape[1:3]
    ongoing_event = False
    ongoing_event_moments = []
    moment_counter = 0
    event_complete_threshold = 5
    event_completion_progress = 0
    event_number = 0
    if not os.path.exists(args.event_dir):
        os.makedirs(args.event_dir)
    else:
        event_number = max([int(e) for e in os.listdir(args.event_dir)]) + 1

    images_path = None
    while True:
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
            prediction_value = np.array(
                model.predict_on_batch(frame_scaled_expanded)).flatten()[0]
            if ongoing_event:
                if prediction_value <= args.threshold:
                    event_completion_progress += 1
                else:
                    event_completion_progress = 0

                if event_completion_progress > event_complete_threshold:
                    ongoing_event = False
                    event_file_path = os.path.join(args.event_dir,
                                                   str(event_number),
                                                   'moments.pickle')
                    with open(event_file_path, 'wb') as event_file:
                        pickle.dump(ongoing_event_moments, event_file)
                    logger.info('##############################')
                    logger.info('End event #{}'.format(event_number))
                    logger.info('##############################')
                    event_completion_progress = 0
                    moment_counter = 0
                    event_number += 1
                    ongoing_event_moments = []
                else:
                    # TODO: factor out
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_path = os.path.join(images_path,
                                            '{}.jpg'.format(moment_counter))
                    image_resized.save(img_path)
                    moment_counter += 1
                    event_moment = {
                        'event_number': event_number,
                        'timestamp': timestamp,
                        'prediction_value': prediction_value,
                        'img_path': img_path,
                    }
                    ongoing_event_moments.append(event_moment)
                    logger.info(
                        'Prediction value: {}'.format(prediction_value))

            elif prediction_value > args.threshold:
                ongoing_event = True
                images_path = os.path.join(args.event_dir, str(event_number),
                                           'images')
                os.makedirs(images_path)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_path = os.path.join(images_path,
                                        '{}.jpg'.format(moment_counter))
                image_resized.save(img_path)
                moment_counter += 1
                event_moment = {
                    'event_number': event_number,
                    'timestamp': timestamp,
                    'prediction_value': prediction_value,
                    'img_path': img_path,
                }
                ongoing_event_moments.append(event_moment)
                logger.info('##############################')
                logger.info('Begin event #{}'.format(event_number))
                logger.info('##############################')
                logger.info('Prediction value: {}'.format(prediction_value))
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
    arg_parser.add_argument('--event-dir',
                            dest='event_dir',
                            required=True,
                            help='Directory to save events in.')
    arg_parser.add_argument('--camera-ip',
                            dest='camera_ip',
                            required=True,
                            help='IP address of the webcam.')
    arg_parser.add_argument(
        '--threshold',
        dest='threshold',
        required=False,
        type=float,
        default=0.5,
        help='Probability threshold for detecting a train.')
    main(arg_parser.parse_args())