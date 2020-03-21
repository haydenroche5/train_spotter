import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from PIL import Image
from io import BytesIO
import numpy as np
import requests
import time
import argparse
from requests.exceptions import RequestException
from datetime import datetime
import pickle
import sys
import logging
import threading
import zmq
import json

from rocheml.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from rocheml.preprocessing.rescalepreprocessor import RescalePreprocessor
from rocheml.preprocessing.resizepreprocessor import ResizePreprocessor
from rocheml.preprocessing.croppreprocessor import CropPreprocessor
from core.detector import Detector
from vision.traindetectionmodel import TrainDetectionModel
from vision.signaldetectionmodel import SignalDetectionModel


def get_camera_img(camera_ip, width, height):
    response = requests.get(
        'http://{}/axis-cgi/jpg/image.cgi?resolution={}x{}'.format(
            camera_ip, width, height),
        timeout=5)
    img = Image.open(BytesIO(response.content))

    return img


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


def eat_updates(zmq_context):
    socket = zmq_context.socket(zmq.SUB)
    socket.connect("inproc://detector")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        data = socket.recv()
        # print(data, flush=True)


def run_detectors(args, zmq_context):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    file_handler = logging.FileHandler(
        os.path.join(args.logging_dir,
                     datetime.now().strftime('%Y%m%d_%H%M%S')) + '.log',
        mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)

    socket = zmq_context.socket(zmq.PUB)
    socket.bind("inproc://detector")

    train_img_width = 1920
    train_img_height = 1080
    train_input_width = 384
    train_input_height = 216
    num_channels = 3

    train_detection_model = TrainDetectionModel.build(
        width=train_input_width,
        height=train_input_height,
        num_channels=num_channels)

    train_resize_preprocessor = ResizePreprocessor(train_input_width,
                                                   train_input_height)
    img_to_array_preprocessor = ImageToArrayPreprocessor('channels_last')
    rescale_preprocessor = RescalePreprocessor(255)

    if args.intersection == 'fourth':
        signal_input_height = 130
        signal_input_width = 130
        signal_box_origins = [(1090, 306), (1218, 515)]
        signal_detection_model = SignalDetectionModel.build(
            width=signal_input_width,
            height=signal_input_height,
            num_channels=num_channels)
        signal_detectors = [
            Detector(signal_detection_model, [
                img_to_array_preprocessor,
                CropPreprocessor(signal_box_origins[0], signal_input_width,
                                 signal_input_height), rescale_preprocessor
            ]),
            Detector(signal_detection_model, [
                img_to_array_preprocessor,
                CropPreprocessor(signal_box_origins[1], signal_input_width,
                                 signal_input_height), rescale_preprocessor
            ])
        ]
    elif args.intersection == 'chestnut':
        signal_input_height = 180
        signal_input_width = 170
        signal_box_origin = (int((315 / 1920.0) * train_img_width), 0)
        signal_detection_model = SignalDetectionModel.build(
            width=signal_input_width,
            height=signal_input_height,
            num_channels=num_channels)
        signal_detectors = [
            Detector(signal_detection_model, [
                img_to_array_preprocessor,
                CropPreprocessor(signal_box_origin, signal_input_width,
                                 signal_input_height), rescale_preprocessor
            ])
        ]
        signal_crop_preprocessor = CropPreprocessor(signal_box_origin,
                                                    [signal_input_width],
                                                    [signal_input_height])
    else:
        raise Exception('Unrecognized intersection: {}.'.format(
            args.intersection))

    logger.info('Threshold: {}.'.format(args.threshold))
    logger.info('Loading models.')
    train_detection_model.load_weights(args.train_model_weights)
    signal_detection_model.load_weights(args.signal_model_weights)
    logger.info('Loading complete.')

    train_detector = Detector(train_detection_model, [
        train_resize_preprocessor, img_to_array_preprocessor,
        rescale_preprocessor
    ])

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

    logger.info('Starting detector.')

    while True:
        try:
            img = get_camera_img(args.camera_ip, train_img_width,
                                 train_img_height)

            signal_prediction_values = []

            for signal_detector in signal_detectors:
                signal_prediction_values.append(signal_detector.detect([img]))

            signal_prediction_value = max(signal_prediction_values).astype(
                float)

            if signal_prediction_value > args.threshold:
                logger.info('Signal is on!')
                logger.info('Signal prediction value: {}'.format(
                    signal_prediction_value))

            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # img_path = os.path.join('/home/pi/signal_check/', '{}.jpg'.format(timestamp))
            # img_to_save = Image.fromarray((signal_detection_input_img[0] * 255).astype(np.uint8))
            # img_to_save.save(img_path)

            train_prediction_value = train_detector.detect([img])

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
                    moment = save_moment(
                        train_resize_preprocessor.preprocess([img])[0],
                        args.event_dir, event_number, moment_counter,
                        train_prediction_value, signal_prediction_value)
                    ongoing_event_moments.append(moment)
                    moment_counter += 1
                    logger.info('Train prediction value: {}'.format(
                        train_prediction_value))
            elif train_prediction_value > args.threshold:
                ongoing_event = True

                moment = save_moment(
                    train_resize_preprocessor.preprocess([img])[0],
                    args.event_dir, event_number, moment_counter,
                    train_prediction_value, signal_prediction_value)
                ongoing_event_moments.append(moment)
                moment_counter += 1
                logger.info('##############################')
                logger.info('Begin event #{}'.format(event_number))
                logger.info('##############################')
                logger.info('Train prediction value: {}'.format(
                    train_prediction_value))

            # if not args.test:
            #     blob = {
            #         "train": train_prediction_value.astype(float),
            #         "signal": signal_prediction_value.astype(float),
            #         "secret": "redacted"
            #     }
            #     r = requests.post(
            #         'https://train-detector.herokuapp.com/update/{}'.format(
            #             args.intersection),
            #         json=blob)

            # socket.send_json({
            #     'train': train_prediction_value.astype(float),
            #     "signal": signal_prediction_value.astype(float)
            # })
            # blob = json.dumps({
            #     'train': train_prediction_value.astype(float),
            #     "signal": signal_prediction_value.astype(float)
            # })
            payload = str.encode(
                f'{train_prediction_value:.8f}, {signal_prediction_value:.8f}')
            socket.send(payload)
            # socket.send(' ' + blob)
            # socket.send_string('Hayden!')
            time.sleep(args.sleep_length)
        except RequestException:
            logger.info('Failed to get image from the webcam. Will try again.')


def main(args):
    zmq_context = zmq.Context()
    detectors_thread = threading.Thread(target=run_detectors,
                                        args=(args, zmq_context),
                                        daemon=True)
    eating_thread = threading.Thread(target=eat_updates,
                                     args=(zmq_context, ),
                                     daemon=True)
    detectors_thread.start()
    eating_thread.start()
    detectors_thread.join()
    eating_thread.join()


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
    arg_parser.add_argument('-g',
                            '--logging-dir',
                            dest='logging_dir',
                            required=True,
                            help='Directory to save logs in.')
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
