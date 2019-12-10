import os
import sys
from os.path import dirname, abspath
root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)
from vision.model import SignalDetectionModel

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def main(args):
    height = 1080
    width = 1920
    num_channels = 3
    scale_factor = 0.2
    # TODO: Make these scaled down images square
    scaled_height = int(height * scale_factor)
    scaled_width = int(width * scale_factor)

    learning_rate = 1e-3

    optimizer = SGD(lr=learning_rate)
    model = SignalDetectionModel.build(width=scaled_width,
                                       height=scaled_height,
                                       num_channels=num_channels)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())

    img_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.25)
    training_generator = img_gen.flow_from_directory(
        args.data_dir,
        target_size=(scaled_height, scaled_width),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True)
    validation_generator = img_gen.flow_from_directory(
        args.data_dir,  # same directory as training data
        target_size=(scaled_height, scaled_width),
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
                                              'signal_detection_model'),
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
        args.output_dir, 'signal_detector_training_history.pickle')
    with open(history_file_path, 'wb') as history_file:
        pickle.dump(H.history, history_file)

    epochs_array = np.arange(args.num_epochs)
    val_loss_line = plt.plot(epochs_array,
                             H.history['val_loss'],
                             label='Validation Loss')
    training_loss_line = plt.plot(epochs_array,
                                  H.history['loss'],
                                  label='Training Loss')
    plt.setp(val_loss_line, linewidth=2.0, marker='+', markersize=10.0)
    plt.setp(training_loss_line, linewidth=2.0, marker='4', markersize=10.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    val_accuracy_line = plt.plot(epochs_array,
                                 H.history['val_accuracy'],
                                 label='Validation Accuracy')
    accuracy_line = plt.plot(epochs_array,
                             H.history['accuracy'],
                             label='Training Accuracy')
    plt.setp(val_accuracy_line, linewidth=2.0, marker='+', markersize=10.0)
    plt.setp(accuracy_line, linewidth=2.0, marker='4', markersize=10.0)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Retrain the signal detection model.')
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
