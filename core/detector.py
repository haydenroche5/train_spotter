from PIL import Image
import PIL
from io import BytesIO
from datetime import datetime
import requests
from requests.exceptions import RequestException
import time
import numpy as np
import logging
import zmq
import os
import base64

from rocheml.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from rocheml.preprocessing.rescalepreprocessor import RescalePreprocessor
from rocheml.preprocessing.resizepreprocessor import ResizePreprocessor
from rocheml.preprocessing.croppreprocessor import CropPreprocessor
from vision.traindetectionmodel import TrainDetectionModel
from vision.signaldetectionmodel import SignalDetectionModel


class Detector:
    def __init__(self, camera_ip, intersection, train_detection_model_weights,
                 signal_detection_model_weights, camera_img_width,
                 camera_img_height, log_file, zmq_endpoint, sleep_length):
        self.camera_ip = camera_ip
        self.camera_img_width = camera_img_width
        self.camera_img_height = camera_img_height
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PUB)
        self.socket.bind('ipc://{}'.format(zmq_endpoint))
        self.sleep_length = sleep_length

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        logging_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging_formatter)
        self.logger.addHandler(file_handler)

        train_input_width = 384
        train_input_height = 216
        num_channels = 3

        self.train_resize_preprocessor = ResizePreprocessor(
            train_input_width, train_input_height)
        self.img_to_array_preprocessor = ImageToArrayPreprocessor(
            'channels_last')
        self.rescale_preprocessor = RescalePreprocessor(255)

        self.train_detection_model = TrainDetectionModel.build(
            width=train_input_width,
            height=train_input_height,
            num_channels=num_channels)

        if intersection == 'fourth':
            signal_input_height = 130
            signal_input_width = 130
            signal_box_origins = [(1090, 306), (1218, 515)]
            self.crop_preprocessors = [
                CropPreprocessor(signal_box_origins[0], signal_input_width,
                                 signal_input_height),
                CropPreprocessor(signal_box_origins[1], signal_input_width,
                                 signal_input_height)
            ]
        elif intersection == 'chestnut':
            signal_input_height = 180
            signal_input_width = 170
            signal_box_origin = (int(
                (315 / 1920.0) * self.camera_img_width), 0)
            self.crop_preprocessors = [
                CropPreprocessor(signal_box_origin, signal_input_width,
                                 signal_input_height)
            ]
        else:
            raise Exception(
                'Unrecognized intersection: {}.'.format(intersection))

        self.signal_detection_model = SignalDetectionModel.build(
            width=signal_input_width,
            height=signal_input_height,
            num_channels=num_channels)

        self.logger.info('Loading models.')
        self.train_detection_model.load_weights(train_detection_model_weights)
        self.signal_detection_model.load_weights(
            signal_detection_model_weights)
        self.logger.info('Loading complete.')

    def run(self):
        self.logger.info('Starting detector.')

        while True:
            try:
                camera_response = requests.get(
                    'http://{}/axis-cgi/jpg/image.cgi?resolution={}x{}'.format(
                        self.camera_ip, self.camera_img_width,
                        self.camera_img_height),
                    timeout=5)
            except RequestException as e:
                self.logger.warn(e)
                self.logger.warn(
                    'Failed to get image from the webcam. Will try again.')
            else:
                img = Image.open(BytesIO(camera_response.content))

                train_img_resized = self.train_resize_preprocessor.preprocess(
                    [img])[0]
                train_img_array = self.img_to_array_preprocessor.preprocess(
                    [train_img_resized])[0]
                train_img_scaled = self.rescale_preprocessor.preprocess(
                    [train_img_array])[0]
                train_prediction_value = np.squeeze(
                    self.train_detection_model.predict_on_batch(
                        np.expand_dims(train_img_scaled, axis=0)))

                signal_prediction_values = []
                signal_img_crops = []
                for crop_pp in self.crop_preprocessors:
                    signal_img_crop = crop_pp.preprocess([img])[0]
                    signal_img_crops.append(signal_img_crop)
                    signal_img_array = self.img_to_array_preprocessor.preprocess(
                        [signal_img_crop])[0]
                    signal_img_scaled = self.rescale_preprocessor.preprocess(
                        [signal_img_array])[0]
                    signal_prediction_values.append(
                        np.squeeze(
                            self.signal_detection_model.predict_on_batch(
                                np.expand_dims(signal_img_scaled, axis=0))))

                signal_prediction_value = max(signal_prediction_values).astype(
                    float)

                train_img_resized_bytes = BytesIO()
                train_img_resized.save(train_img_resized_bytes, format="JPEG")
                train_img_resized_payload = base64.b64encode(
                    train_img_resized_bytes.getvalue())

                signal_img_crop_payloads = []
                for signal_img_crop in signal_img_crops:
                    signal_img_crop_bytes = BytesIO()
                    signal_img_crop.save(signal_img_crop_bytes, format="JPEG")
                    signal_img_crop_payloads.append(
                        base64.b64encode(signal_img_crop_bytes.getvalue()))

                predictions_payload = str.encode('{:.8f}, {:.8f}'.format(
                    train_prediction_value, signal_prediction_value))

                self.socket.send_multipart([
                    predictions_payload, train_img_resized_payload,
                    *signal_img_crop_payloads
                ])

                time.sleep(self.sleep_length)
