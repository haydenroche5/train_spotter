import os
from vision.traindetectionmodel import TrainDetectionModel
import argparse
from datetime import datetime
import json
import math
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    height = 216
    width = 384
    num_channels = 3

    if config['optimizer'] == 'SGD':
        optimizer = SGD(lr=config['learning_rate'],
                        decay=config['learning_rate'] / config['epochs'],
                        momentum=config['momentum'])
    else:
        raise Exception('Unsupported optimizer: {}.'.format(
            config['optimizer']))

    model = TrainDetectionModel.build(width=width,
                                      height=height,
                                      num_channels=num_channels)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())

    img_gen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=config['validation_split'],
        width_shift_range=config['width_shift_range'],
        height_shift_range=config['height_shift_range'],
        fill_mode=config['fill_mode'],
        horizontal_flip=config['horizontal_flip'])
    training_generator = img_gen.flow_from_directory(
        args.data_dir,
        target_size=(height, width),
        batch_size=config['batch_size'],
        class_mode='binary',
        subset='training',
        shuffle=True)
    validation_generator = img_gen.flow_from_directory(
        args.data_dir,
        target_size=(height, width),
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
        ModelCheckpoint(os.path.join(output_sub_dir,
                                     'model.{epoch:02d}-{val_loss:.4f}.hdf5'),
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
        description='Train the train detection model.')
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
