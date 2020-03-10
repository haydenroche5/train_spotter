from PIL import Image
from io import BytesIO
import numpy as np
import requests
import time
import argparse
from vision.traindetectionmodel import TrainDetectionModel
from vision.signaldetectionmodel import SignalDetectionModel
from requests.exceptions import RequestException
from datetime import datetime
import pickle
import os
import sys


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


def prepare_signal_img_chestnut(model, img):
    model_img_height, model_img_width = model.layers[0].input_shape[1:3]
    image_array = np.array(img)
    img_width = image_array.shape[1]
    signal_x = int((315 / 1920.0) * img_width)
    signal_y = 0
    cropped_img = image_array[signal_y:signal_y +
                              model_img_height, signal_x:signal_x +
                              model_img_width]
    cropped_img_scaled = cropped_img / 255.0
    cropped_img_scaled_expanded = np.expand_dims(cropped_img_scaled, axis=0)

    return cropped_img_scaled_expanded


def prepare_signal_imgs_fourth(model, img):
    signal_xs = [1090, 1218]
    signal_ys = [306, 515]

    input_img_height, input_img_width = model.layers[0].input_shape[1:3]
    cropped_imgs = []
    for x, y in zip(signal_xs, signal_ys):
        image_array = np.array(img)
        cropped_img = image_array[y:y + input_img_height, x:x +
                                  input_img_width]
        cropped_img_scaled = cropped_img / 255.0
        cropped_img_scaled_expanded = np.expand_dims(cropped_img_scaled,
                                                     axis=0)
        cropped_imgs.append(cropped_img_scaled_expanded)

    return cropped_imgs


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
    train_img_width = 1920
    train_img_height = 1080
    train_input_height = 384
    train_input_width = 216
    num_channels = 3

    if args.intersection == 'fourth':
        signal_input_height = 130
        signal_input_width = 130
    elif args.intersection == 'chestnut':
        signal_input_height = 180
        signal_input_width = 170
    else:
        raise Exception('Unrecognized intersection: {}.'.format(
            args.intersection))

    print('Threshold: {}.'.format(args.threshold))

    print('Loading models.')
    train_detection_model = TrainDetectionModel.build(
        width=train_input_width,
        height=train_input_height,
        num_channels=num_channels)
    signal_detection_model = SignalDetectionModel.build(
        width=signal_input_width,
        height=signal_input_height,
        num_channels=num_channels)
    train_detection_model.load_weights(args.train_model_weights)
    signal_detection_model.load_weights(args.signal_model_weights)
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

    print('Starting detector.')

    while True:
        try:
            img = get_camera_img(args.camera_ip, train_img_width,
                                 train_img_height)
            img_resized = img.resize((train_input_width, train_input_height))
            train_detection_input_img = prepare_img_for_train_detection(
                img_resized)

            if args.intersection == 'fourth':
                signal_detection_input_imgs = prepare_signal_imgs_fourth(
                    signal_detection_model, img)
                signal_prediction_values = []

                for signal_img in signal_detection_input_imgs:
                    signal_prediction_value = np.array(
                        signal_detection_model.predict_on_batch(
                            signal_img)).flatten()[0]
                    signal_prediction_values.append(signal_prediction_value)

                    if signal_prediction_value > args.threshold:
                        print('Signal is on!')
                        print('Signal prediction value: {}'.format(
                            signal_prediction_value))

                signal_prediction_value = max(signal_prediction_values).astype(
                    float)
            elif args.intersection == 'chestnut':
                signal_detection_input_img = prepare_signal_img_chestnut(
                    signal_detection_model, img)
                signal_prediction_value = np.array(
                    signal_detection_model.predict_on_batch(
                        signal_detection_input_img)).flatten()[0]
                # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                # img_path = os.path.join('/home/pi/signal_check/', '{}.jpg'.format(timestamp))
                # img_to_save = Image.fromarray((signal_detection_input_img[0] * 255).astype(np.uint8))
                # img_to_save.save(img_path)

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

            if not args.test:
                blob = {
                    "train": train_prediction_value.astype(float),
                    "signal": signal_prediction_value.astype(float),
                    "secret": "redacted"
                }
                r = requests.post(
                    'https://train-detector.herokuapp.com/update/{}'.format(
                        args.intersection),
                    json=blob)

            time.sleep(args.sleep_length)
        except RequestException:
            print('Failed to get image from the webcam. Will try again.')

        if args.test:
            sys.stdout.flush()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Run the train detector.')
    arg_parser.add_argument(
        '-i',
        '--intersection',
        dest='intersection',
        required=True,
        help=
        'The intersection that the camera is pointed at. One of \'chestnut\' or \'fourth\'.'
    )
    arg_parser.add_argument('--test', action='store_true')
    arg_parser.add_argument('-t',
                            '--train-model-weights',
                            dest='train_model_weights',
                            required=True,
                            help='Path to the train detection model weights.')
    arg_parser.add_argument('-s',
                            '--signal-model-weights',
                            dest='signal_model_weights',
                            required=True,
                            help='Path to the signal detection model weights.')
    arg_parser.add_argument('-e',
                            '--event-dir',
                            dest='event_dir',
                            required=True,
                            help='Directory to save events in.')
    arg_parser.add_argument('-c',
                            '--camera-ip',
                            dest='camera_ip',
                            required=True,
                            help='IP address of the webcam.')
    arg_parser.add_argument(
        '-r',
        '--threshold',
        dest='threshold',
        required=False,
        type=float,
        default=0.5,
        help='Probability threshold for detecting a train.')
    arg_parser.add_argument('-l',
                            '--sleep-length',
                            dest='sleep_length',
                            required=False,
                            type=float,
                            default=3.0,
                            help='Number of seconds to sleep between updates.')
    main(arg_parser.parse_args())