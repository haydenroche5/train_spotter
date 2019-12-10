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


def get_camera_img(camera_ip, width, height):
    response = requests.get(
        'http://{}/axis-cgi/jpg/image.cgi?resolution={}x{}'.format(
            args.camera_ip, width, height),
        timeout=5)
    img = Image.open(BytesIO(response.content))

    return img


def prepare_img_for_train_detection(img, model):
    input_img_height, input_img_width = model.layers[0].input_shape[1:3]
    img_resized = img.resize((input_img_width, input_img_height))
    image_array = np.array(image_resized)
    image_array_scaled = image_array / 255.0
    image_array_scaled_expanded = np.expand_dims(image_array_scaled, axis=0)

    return image_array_scaled_expanded


def prepare_imgs_for_signal_detection(img, model):
    signal_xs = [1090, 1218]
    signal_ys = [306, 515]

    input_img_height, input_img_width = model.layers[0].input_shape[1:3]
    cropped_imgs = []
    for x, y in zip(signal_xs, signal_ys):
        cropped_img = img[y:y + input_img_height, x:x + input_img_width]
        image_array = np.array(cropped_img)
        image_array_scaled = cropped_img / 255.0
        image_array_scaled_expanded = np.expand_dims(image_array_scaled,
                                                     axis=0)
        cropped_imgs.append(image_array_scaled_expanded)

    return cropped_imgs


def save_moment(event_dir, event_number, moment_number,
                train_prediction_value):
    images_path = os.path.join(event_dir, str(event_number), 'images')
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_path = os.path.join(images_path, '{}.jpg'.format(moment_number))
    image_resized.save(img_path)
    moment = {
        'event_number': event_number,
        'timestamp': timestamp,
        'train_prediction_value': train_prediction_value,
        'img_path': img_path,
    }

    return moment


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

    train_detection_model = load_model(
        os.path.join(args.model_dir, 'train_detection', 'model'))
    signal_detection_model = load_model(
        os.path.join(args.model_dir, 'signal_detection', 'model'))

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
            img = get_camera_img(args.camera_ip, width, height)
            train_detection_input_img = prepare_img_for_train_detection(img)
            signal_detection_input_imgs = prepare_imgs_for_signal_detection(
                img)

            for img in signal_detection_input_imgs:
                signal_prediction_value = np.array(
                    model.predict_on_batch(img)).flatten()[0]
                logger.info('Signal prediction value: {}'.format(
                    signal_prediction_value))

            train_prediction_value = np.array(
                model.predict_on_batch(train_detection_input_img)).flatten()[0]

            if ongoing_event:
                if train_prediction_value <= args.threshold:
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
                    moment = save_moment(args.event_dir, event_number,
                                         moment_counter,
                                         train_prediction_value)
                    ongoing_event_moments.append(moment)
                    logger.info('Train prediction value: {}'.format(
                        train_prediction_value))

            elif train_prediction_value > args.threshold:
                ongoing_event = True
                moment = save_moment(args.event_dir, event_number,
                                     moment_counter, train_prediction_value)
                ongoing_event_moments.append(moment)
                logger.info('##############################')
                logger.info('Begin event #{}'.format(event_number))
                logger.info('##############################')
                logger.info('Train prediction value: {}'.format(
                    train_prediction_value))
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
                            help='Directory containing models.')
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