import zmq
import logging
import requests


class WebPublisher:
    def __init__(self, test_mode, intersection, zmq_context, zmq_endpoint,
                 log_file):
        self.test_mode = test_mode

        if intersection not in ['fourth', 'chestnut']:
            raise Exception('Invalid intersection: {}.'.format(intersection))

        self.intersection = intersection

        self.socket = zmq_context.socket(zmq.SUB)
        self.socket.connect(f'inproc://{zmq_endpoint}')
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        logging_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging_formatter)
        self.logger.addHandler(file_handler)

    def run(self):
        while True:
            print('buh 0', flush=True)
            predictions, _ = self.socket.recv_multipart()
            print('buh 1', flush=True)
            train_prediction_value, signal_prediction_value = [
                float(val) for val in predictions.decode().split(', ')
            ]

            if self.test_mode:
                print('buh 2', flush=True)
                self.logger.info(
                    'Test mode. Would have published: train: {}, signal: {}.'.
                    format(train_prediction_value, signal_prediction_value))
            else:
                blob = {
                    "train": train_prediction_value,
                    "signal": signal_prediction_value,
                    "secret": "redacted"
                }
                r = requests.post(
                    'https://train-detector.herokuapp.com/update/{}'.format(
                        self.intersection),
                    json=blob)
