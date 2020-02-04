from PIL import Image
from io import BytesIO
import numpy as np
import requests
import time
import json
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
            camera_ip, width, height),
        timeout=5)
    img = Image.open(BytesIO(response.content))

    return img


def prepare_img_for_train_detection(img):
    image_array = np.array(img)
    image_array_scaled = image_array / 255.0
    image_array_scaled_expanded = np.expand_dims(image_array_scaled, axis=0)

    return image_array_scaled_expanded


def prepare_img_for_signal_detection(model, img):
    signal_x = 470
    signal_y = 0
    input_img_height, input_img_width = model.layers[0].input_shape[1:3]
    image_array = np.array(img)
    cropped_img = image_array[signal_y:signal_y +
                              input_img_height, signal_x:signal_x +
                              input_img_width]
    cropped_img_scaled = cropped_img / 255.0
    cropped_img_scaled_expanded = np.expand_dims(cropped_img_scaled, axis=0)

    return cropped_img_scaled_expanded


def save_moment(img, event_dir, event_number, moment_number,
                train_prediction_value, signal_prediction_value):
    images_path = os.path.join(event_dir, str(event_number), 'images')
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_path = os.path.join(images_path, '{}.jpg'.format(moment_number))
    img.save(img_path)
    moment = {
        'event_number': event_number,
        'timestamp': timestamp,
        'train_prediction_value': train_prediction_value,
        'img_path': img_path,
        'signal_prediction_value': signal_prediction_value,
    }

    return moment


def main(args):
    width = 1920
    height = 1080
    print('THRESHOLD: {}.'.format(args.threshold))
    print('Loading models.')
    train_detection_model = load_model(
        os.path.join(args.model_dir, 'train_detection', 'model'))
    signal_detection_model = load_model(
        os.path.join(args.model_dir, 'signal_detection', 'model'))
    print('Loading complete.')

    ongoing_event = False
    ongoing_event_moments = []
    moment_counter = 0
    event_complete_threshold = 5
    event_completion_progress = 0
    event_number = 0
    if not os.path.exists(args.event_dir):
        os.makedirs(args.event_dir)
    else:
        if os.listdir(args.event_dir):
            event_number = max([int(e)
                                for e in os.listdir(args.event_dir)]) + 1

    print('Starting detector')
    while True:
        try:
            img = get_camera_img(args.camera_ip, width, height)
            input_img_height, input_img_width = train_detection_model.layers[
                0].input_shape[1:3]
            img_resized = img.resize((input_img_width, input_img_height))
            train_detection_input_img = prepare_img_for_train_detection(
                img_resized)
            signal_detection_input_img = prepare_img_for_signal_detection(
                signal_detection_model, img)

            signal_prediction_value = np.array(
                signal_detection_model.predict_on_batch(
                    signal_detection_input_img)).flatten()[0]
            if signal_prediction_value > args.threshold:
                print('Signal is on!')
                print('Signal prediction value: {}'.format(
                    signal_prediction_value))

            train_prediction_value = np.array(
                train_detection_model.predict_on_batch(
                    train_detection_input_img)).flatten()[0]

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
                    print('##############################')
                    print('End event #{}'.format(event_number))
                    print('##############################')
                    event_completion_progress = 0
                    moment_counter = 0
                    event_number += 1
                    ongoing_event_moments = []
                else:
                    moment = save_moment(img_resized, args.event_dir,
                                         event_number, moment_counter,
                                         train_prediction_value,
                                         signal_prediction_value)
                    ongoing_event_moments.append(moment)
                    moment_counter += 1
                    print('Train prediction value: {}'.format(
                        train_prediction_value))

            elif train_prediction_value > args.threshold:
                ongoing_event = True
                moment = save_moment(img_resized, args.event_dir, event_number,
                                     moment_counter, train_prediction_value,
                                     signal_prediction_value)
                ongoing_event_moments.append(moment)
                moment_counter += 1
                print('##############################')
                print('Begin event #{}'.format(event_number))
                print('##############################')
                print('Train prediction value: {}'.format(
                    train_prediction_value))

            # blob = {
            #     "train": train_prediction_value.astype(float),
            #     "secret": "redacted"
            # }
            # r = requests.post('https://train-detector.herokuapp.com/update',
            #                   json=blob)
            time.sleep(3)
        except ConnectionError:
            print(
                'Timed out trying to get an image from the webcam. Will try again.'
            )


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Run the train detector.')
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