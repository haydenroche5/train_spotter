import argparse
from datetime import datetime
import multiprocessing
import zmq
import sys
import traceback
import os

from core.detector import Detector
from core.eventtracker import EventTracker
from core.webpublisher import WebPublisher


def run_event_tracker(args, zmq_endpoint, log_file):
    try:
        event_tracker = EventTracker(args.threshold, zmq_endpoint,
                                     args.event_dir, log_file)
        event_tracker.run()
    except:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()


def run_detector(args, zmq_endpoint, log_file):
    camera_img_width = 1920
    camera_img_height = 1080

    try:
        detector = Detector(args.camera_ip, args.intersection,
                            args.train_model_weights,
                            args.signal_model_weights, camera_img_width,
                            camera_img_height, log_file, zmq_endpoint,
                            args.sleep_length)
        detector.run()
    except:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()


def run_web_publisher(args, zmq_endpoint, log_file):
    try:
        web_publisher = WebPublisher(args.test, args.intersection,
                                     zmq_endpoint, log_file, args.api_secret)
        web_publisher.run()
    except:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()


def main(args):
    zmq_endpoint = 'detector'
    log_file = os.path.join(args.logging_dir,
                            datetime.now().strftime('%Y%m%d_%H%M%S') + '.log')
    multiprocessing.set_start_method('spawn')

    detector_process = multiprocessing.Process(target=run_detector,
                                               args=(args, zmq_endpoint,
                                                     log_file),
                                               daemon=True)
    event_tracker_process = multiprocessing.Process(target=run_event_tracker,
                                                    args=(args, zmq_endpoint,
                                                          log_file),
                                                    daemon=True)
    web_publisher_process = multiprocessing.Process(target=run_web_publisher,
                                                    args=(args, zmq_endpoint,
                                                          log_file),
                                                    daemon=True)

    detector_process.start()
    event_tracker_process.start()
    web_publisher_process.start()

    detector_process.join()
    event_tracker_process.join()
    web_publisher_process.join()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Run the train spotter.')
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
    arg_parser.add_argument(
        '--api-secret',
        dest='api_secret',
        required=True,
        help='API secret for updating the server with predictions.')
    main(arg_parser.parse_args())
