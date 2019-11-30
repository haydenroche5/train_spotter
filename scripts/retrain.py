import logging
import argparse
from utils.hashing import get_dir_hash
from detection.model import Model
import os.path
import subprocess
from subprocess import CalledProcessError
import shlex
from datetime import date
import sys


class StdLogger(object):
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        self.logger.info(message)

    def flush(self):
        pass


def train_model(data_dir_path, output_dir_path, num_epochs, batch_size,
                patience):
    height = 1080
    width = 1920
    num_channels = 3
    scale_factor = 0.2
    scaled_height = int(height * scale_factor)
    scaled_width = int(width * scale_factor)
    model = Model(scaled_height, scaled_width, num_channels, data_dir_path,
                  batch_size)
    model.train(num_epochs, patience, output_dir_path)


def get_date():
    return date.today().strftime('%Y%m%d')


def stash_local_changes():
    command = 'git stash save --include-untracked "retrain_{}"'.format(
        get_date())
    result = subprocess.run(shlex.split(command), capture_output=True)
    if 'No local changes to save' in result.stdout.decode():
        return False
    return True


def pop_local_changes():
    command = 'git stash pop'
    subprocess.check_call(shlex.split(command))


def pull_latest_data():
    command = 'git pull origin master --rebase'
    subprocess.check_call(shlex.split(command))


def push_latest_model(output_dir_path):
    command = 'git add {}'.format(output_dir_path)
    subprocess.check_call(shlex.split(command))
    command = 'git commit -m "{} retraining."'.format(get_date())
    subprocess.check_call(shlex.split(command))
    current_version_file_path = os.path.join(output_dir_path,
                                             'current_version.txt')
    with open(current_version_file_path, "r+") as file:
        current_version = int(file.read())
        file.seek(0)
        new_version = current_version + 1
        file.write(str(new_version) + '\n')
        file.truncate()
    command = 'git tag -a v{version} -m "New model version: {version}"'.format(
        version=new_version)
    subprocess.check_call(shlex.split(command))
    command = 'git push origin master'
    subprocess.check_call(shlex.split(command))


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_dir_path = os.path.join(args.output_dir, 'log/')
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    retrain_log_path = os.path.join(args.output_dir,
                                    'log/retrain_{}.log'.format(get_date()))
    fh = logging.FileHandler(retrain_log_path, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                  datefmt='%m-%d-%Y %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sys.stdout = StdLogger(logger)
    sys.stderr = StdLogger(logger)

    saved = stash_local_changes()
    try:
        pull_latest_data()
        if not args.force:
            current_data_hash_file_path = os.path.join(
                args.output_dir, 'current_data_hash.md5')
            with open(current_data_hash_file_path, 'r') as file:
                current_hash = file.read()
            new_hash = get_dir_hash(args.data_dir)
            logger.info('Current hash: {}'.format(current_hash))
            logger.info('New hash: {}'.format(new_hash))
            if new_hash != current_hash:
                logger.info('Data has changed, re-training.')
                with open(current_data_hash_file_path, 'w') as file:
                    file.write(new_hash)
                train_model(args.data_dir, args.output_dir, args.num_epochs,
                            args.batch_size, args.patience)
                push_latest_model(args.output_dir)
            else:
                logger.info('Data has not changed, not re-training.')
        else:
            logger.info('Forcibly retraining.')
            train_model(args.data_dir, args.output_dir, args.num_epochs,
                        args.batch_size, args.patience)
            push_latest_model(args.output_dir)
    finally:
        if saved:
            pop_local_changes()
    logger.info('Retraining complete.')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Retrain the train detection model.')
    arg_parser.add_argument('--force', action='store_true')
    arg_parser.add_argument('--data-dir',
                            dest='data_dir',
                            required=True,
                            help='Directory containing training data.')
    arg_parser.add_argument('--output-dir',
                            dest='output_dir',
                            required=True,
                            help='Directory to save training results in.')
    arg_parser.add_argument('--num-epochs',
                            dest='num_epochs',
                            default=50,
                            type=int,
                            help='Number of epochs to train for.')
    arg_parser.add_argument(
        '--batch-size',
        dest='batch_size',
        default=16,
        type=int,
        help='Number of images to process in one training step.')
    arg_parser.add_argument(
        '--patience',
        dest='patience',
        default=3,
        help=
        'Max number of epochs to train for without validation loss improvement.'
    )
    main(arg_parser.parse_args())
