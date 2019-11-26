# import logging
import argparse
from utils.hashing import get_dir_hash
from detection.model import Model
import os.path


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


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, 'log/'))

    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)
    # retrain_log_path = os.path.join(args.output_dir, 'log/retrain.log')
    # fh = logging.FileHandler(retrain_log_path, mode='w')
    # fh.setLevel(logging.DEBUG)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.ERROR)
    # formatter = logging.Formatter('[%(asctime)s] %(message)s',
    #                               datefmt='%m-%d-%Y %H:%M:%S')
    # fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.addHandler(ch)

    if not args.force:
        current_data_hash_file = 'current_data_hash.md5'
        current_hash = ''
        with open(current_data_hash_file, 'r') as file:
            current_hash = file.read()
        new_hash = get_dir_hash(args.data_dir)
        print('Current hash: {}'.format(current_hash))
        print('New hash: {}'.format(new_hash))
        if new_hash != current_hash:
            print('Data has changed, re-training.')
            with open(current_data_hash_file, 'w') as file:
                file.write(new_hash)
            train_model(args.data_dir, args.output_dir, args.num_epochs,
                        args.batch_size, args.patience)
        else:
            print('Data has not changed, not re-training.')
    else:
        print('Forcibly retraining.')
        train_model(args.data_dir, args.output_dir, args.num_epochs,
                    args.batch_size, args.patience)


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
