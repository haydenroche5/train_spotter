import os
import logging
from PIL import Image
from io import BytesIO
from datetime import datetime
import zmq
import pickle
import base64


class EventTracker:
    def __init__(self,
                 threshold,
                 zmq_endpoint,
                 event_dir,
                 log_file,
                 save_signal_moments=False):
        self.threshold = threshold
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.SUB)
        self.socket.connect('ipc://{}'.format(zmq_endpoint))
        self.socket.setsockopt_string(zmq.SUBSCRIBE,
                                      "")  # TODO: see if can use single quotes
        self.event_dir = event_dir

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        logging_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        file_handler = logging.FileHandler(log_file)

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging_formatter)
        self.logger.addHandler(file_handler)

        self.save_signal_moments = save_signal_moments

        self.logger.info('Threshold: {}.'.format(self.threshold))

    def save_moment(self, train_img, signal_imgs, event_number, moment_number,
                    train_prediction_value, signal_prediction_value):
        images_path = os.path.join(self.event_dir, str(event_number), 'images')

        if not os.path.exists(images_path):
            os.makedirs(images_path)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        train_img_path = os.path.join(images_path,
                                      'train_{}.jpg'.format(moment_number))
        train_img.save(train_img_path)

        signal_img_paths = []
        for i, signal_img in enumerate(signal_imgs):
            signal_img_path = os.path.join(
                images_path, 'signal_{}_{}.jpg'.format(moment_number, i))
            signal_img.save(signal_img_path)
            signal_img_paths.append(signal_img_path)

        moment = {
            'event_number': event_number,
            'timestamp': timestamp,
            'train_prediction_value': train_prediction_value,
            'train_img_path': train_img_path,
            'signal_img_paths': signal_img_paths,
            'signal_prediction_value': signal_prediction_value,
        }

        return moment

    def run(self):
        ongoing_event = False
        ongoing_event_moments = []
        moment_counter = 0
        event_complete_threshold = 5
        event_completion_progress = 0
        event_number = 0

        if not os.path.exists(self.event_dir):
            os.makedirs(self.event_dir)
        else:
            if os.listdir(self.event_dir):
                event_number = max(
                    [int(e) for e in os.listdir(self.event_dir)]) + 1

        while True:
            payloads = self.socket.recv_multipart()
            predictions = payloads[0]
            train_img_b64 = payloads[1]
            signal_imgs_b64 = payloads[2:]

            train_prediction_value, signal_prediction_value = [
                float(val) for val in predictions.decode().split(', ')
            ]
            train_img = Image.open(BytesIO(base64.b64decode(train_img_b64)))
            signal_imgs = [
                Image.open(BytesIO(base64.b64decode(signal_img_b64)))
                for signal_img_b64 in signal_imgs_b64
            ]

            if signal_prediction_value > self.threshold:
                self.logger.info('Signal is on. Prediction value: {}.'.format(
                    signal_prediction_value))

                if self.save_signal_moments:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    signal_img_path = 'signal_{}.jpg'.format(timestamp)
                    signal_imgs[0].save(signal_img_path)

            if ongoing_event:
                if train_prediction_value <= self.threshold:
                    event_completion_progress += 1
                else:
                    event_completion_progress = 0

                if event_completion_progress > event_complete_threshold:
                    ongoing_event = False
                    event_file_path = os.path.join(self.event_dir,
                                                   str(event_number),
                                                   'moments.pickle')

                    with open(event_file_path, 'wb') as event_file:
                        pickle.dump(ongoing_event_moments, event_file)
                    self.logger.info('##############################')
                    self.logger.info('End event #{}'.format(event_number))
                    self.logger.info('##############################')
                    event_completion_progress = 0
                    moment_counter = 0
                    event_number += 1
                    ongoing_event_moments = []
                else:
                    moment = self.save_moment(train_img, signal_imgs,
                                              event_number, moment_counter,
                                              train_prediction_value,
                                              signal_prediction_value)
                    ongoing_event_moments.append(moment)
                    moment_counter += 1
                    self.logger.info('Train prediction value: {}'.format(
                        train_prediction_value))
            elif train_prediction_value > self.threshold:
                ongoing_event = True

                moment = self.save_moment(train_img, signal_imgs, event_number,
                                          moment_counter,
                                          train_prediction_value,
                                          signal_prediction_value)
                ongoing_event_moments.append(moment)
                moment_counter += 1
                self.logger.info('##############################')
                self.logger.info('Begin event #{}'.format(event_number))
                self.logger.info('##############################')
                self.logger.info('Train prediction value: {}'.format(
                    train_prediction_value))