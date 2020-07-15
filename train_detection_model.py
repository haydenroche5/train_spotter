import os
from vision.signaldetectionmodel import SignalDetectionModel
from vision.traindetectionmodel import TrainDetectionModel
import argparse
from datetime import datetime
import json
import math
import cv2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.model_type == 'signal':
        model = SignalDetectionModel.build(width=config['width'],
                                           height=config['height'],
                                           num_channels=config['num_channels'])
    elif args.model_type == 'train':
        model = TrainDetectionModel.build(width=config['width'],
                                          height=config['height'],
                                          num_channels=config['num_channels'])
    else:
        raise Exception('Unsupported model type: {}.'.format(args.model_type))

    if config['optimizer'] == 'SGD':
        optimizer = SGD(lr=config['learning_rate'],
                        decay=config['learning_rate'] / config['epochs'],
                        momentum=config['momentum'])
    else:
        raise Exception('Unsupported optimizer: {}.'.format(
            config['optimizer']))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())

    if 'width_shift_range' in config:
        width_shift_range = config['width_shift_range']
    else:
        width_shift_range = 0.0

    if 'height_shift_range' in config:
        height_shift_range = config['height_shift_range']
    else:
        height_shift_range = 0.0

    if 'horizontal_flip' in config:
        horizontal_flip = config['horizontal_flip']
    else:
        horizontal_flip = False

    if 'rotation_range' in config:
        rotation_range = config['rotation_range']
    else:
        rotation_range = 0

    img_gen = ImageDataGenerator(rescale=1. / 255,
                                 validation_split=config['validation_split'],
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range,
                                 fill_mode='nearest',
                                 horizontal_flip=horizontal_flip,
                                 rotation_range=rotation_range)
    training_generator = img_gen.flow_from_directory(
        args.data_dir,
        target_size=(config['height'], config['width']),
        batch_size=config['batch_size'],
        class_mode='binary',
        subset='training',
        shuffle=True)
    validation_generator = img_gen.flow_from_directory(
        args.data_dir,
        target_size=(config['height'], config['width']),
        batch_size=config['batch_size'],
        class_mode='binary',
        subset='validation',
        shuffle=False)

    num_training_samples = len(training_generator.labels)
    num_validation_samples = len(validation_generator.labels)

    output_sub_dir = os.path.join(args.output_dir,
                                  datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_sub_dir)

    callbacks = [
        EarlyStopping(monitor='val_loss',
                      patience=config['patience'],
                      restore_best_weights=True,
                      verbose=True),
        ModelCheckpoint(filepath=os.path.join(
            output_sub_dir, 'model.{epoch:02d}-{val_loss:.4f}.hdf5'),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=True),
        CSVLogger(os.path.join(output_sub_dir, 'epochs.csv'))
    ]

    H = model.fit_generator(training_generator,
                            steps_per_epoch=math.ceil(num_training_samples /
                                                      config['batch_size']),
                            epochs=config['epochs'],
                            callbacks=callbacks,
                            validation_data=validation_generator,
                            validation_steps=math.ceil(num_validation_samples /
                                                       config['batch_size']))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Train a detection model.')
    arg_parser.add_argument(
        '--model-type',
        dest='model_type',
        required=True,
        help='Model type to train. One of [train, signal].')
    arg_parser.add_argument('--data-dir',
                            dest='data_dir',
                            required=True,
                            help='Directory containing training data.')
    arg_parser.add_argument('--output-dir',
                            dest='output_dir',
                            required=True,
                            help='Directory to save training results in.')
    arg_parser.add_argument('--config',
                            dest='config',
                            required=True,
                            help='Path to a JSON config file.')
    main(arg_parser.parse_args())
