import os
import sys
from os.path import dirname, abspath
root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)
from vision.model import TrainDetectionModel

import cv2
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


def main(args):
    raw_height = 1080
    raw_width = 1920
    num_channels = 3
    scale_factor = 0.2
    height = int(raw_height * scale_factor)
    width = int(raw_width * scale_factor)
    learning_rate = 1e-2
    decay = learning_rate / args.num_epochs
    momentum = 0.9

    optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum)
    model = TrainDetectionModel.build(width=width,
                                      height=height,
                                      num_channels=num_channels)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())

    img_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.15)
    training_generator = img_gen.flow_from_directory(
        args.data_dir,
        target_size=(height, width),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True)
    validation_generator = img_gen.flow_from_directory(
        args.data_dir,
        target_size=(height, width),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False)

    num_training_samples = len(training_generator.labels)
    num_validation_samples = len(validation_generator.labels)

    callbacks = [
        EarlyStopping(monitor='val_loss',
                      patience=args.patience,
                      restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join(args.output_dir,
                                              'train_detection_model'),
                        monitor='val_loss',
                        save_best_only=True)
    ]

    H = model.fit_generator(
        training_generator,
        steps_per_epoch=math.ceil(num_training_samples / args.batch_size),
        epochs=args.num_epochs,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=math.ceil(num_validation_samples / args.batch_size))

    history_file_path = os.path.join(
        args.output_dir, 'train_detection_model_training_history.pickle')
    with open(history_file_path, 'wb') as history_file:
        pickle.dump(H.history, history_file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Train the train detection model.')
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
        default=32,
        type=int,
        help='Number of images to process in one training step.')
    arg_parser.add_argument(
        '--patience',
        dest='patience',
        default=3,
        type=int,
        help=
        'Max number of epochs to train for without validation loss improvement.'
    )
    main(arg_parser.parse_args())
