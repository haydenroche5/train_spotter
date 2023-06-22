import zmq
import logging
import requests


class WebPublisher:
    def __init__(self, test_mode, intersection, zmq_endpoint, log_file,
                 api_secret):
        self.test_mode = test_mode

        if intersection not in ['fourth', 'chestnut']:
            raise Exception('Invalid intersection: {}.'.format(intersection))

        self.intersection = intersection

        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.SUB)
        self.socket.connect('ipc://{}'.format(zmq_endpoint))
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        logging_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging_formatter)
        self.logger.addHandler(file_handler)

        self.api_secret = api_secret

    def run(self):
        while True:
            try:
                predictions = self.socket.recv_multipart()
            except zmq.ZMQError as e:
                self.logger.warn("ZMQError: {}".format(str(e)))
                continue

            if len(predictions) == 0:
                self.logger.warn("Empty predictions list received on ZMQ socket from detector.")
                continue

            train_prediction_value, signal_prediction_value = [
                float(val) for val in predictions[0].decode().split(', ')
            ]

            if self.test_mode:
                self.logger.info(
                    'Test mode. Would have published: train: {}, signal: {}.'.
                    format(train_prediction_value, signal_prediction_value))
            else:
                blob = {
                    "train": train_prediction_value,
                    "signal": signal_prediction_value,
                    "secret": self.api_secret
                }
                base_urls = [
                    "https://train.cohub.com/api",
                ]
                for base_url in base_urls:
                    try:
                        r = requests.post(
                            '{}/update/{}'.
                            format(base_url, self.intersection),
                            json=blob)
                        r.raise_for_status()
                    except requests.exceptions.HTTPError as err:
                        self.logger.warn("HTTP error ({}): {}".format(base_url, err))
                    except requests.exceptions.ConnectionError as err:
                        self.logger.warn("Connection error ({}): {}".format(base_url, err))
                    except requests.exceptions.Timeout as err:
                        self.logger.warn("Timeout error ({}): {}".format(base_url, err))
                    except requests.exceptions.RequestException as err:
                        self.logger.warn("Request error ({}): {}".format(base_url, err))
