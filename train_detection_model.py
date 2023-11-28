import os
from vision.signaldetectionmodel import SignalDetectionModel
from vision.traindetectionmodel import TrainDetectionModel
import argparse
from datetime import datetime
import json
import math
import cv2
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

# from tensorflow.keras.applications import MobileNetV3Small
# from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

class ConfusionMatrixCallback(Callback):
    def __init__(self, validation_generator):
        super(ConfusionMatrixCallback, self).__init__()
        self.validation_generator = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        # Generate predictions on the validation set
        self.validation_generator.reset()
        predictions = self.model.predict_generator(self.validation_generator)

        # Convert predictions and true labels to class indices
        true_labels = self.validation_generator.classes

        # Set a threshold for classification (e.g., 0.5)
        # threshold = 0.5
        threshold = 0.9

        # Convert probabilities to class indices based on the threshold
        predicted_labels = (predictions > threshold).astype(int)
        predicted_labels = np.squeeze(predicted_labels)

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Print confusion matrix as a table
        cm_df = pd.DataFrame(cm, index=self.validation_generator.class_indices, columns=self.validation_generator.class_indices)
        print(f'\nConfusion Matrix - Epoch {epoch}:\n{cm_df}')

        missclass_indices = np.where(np.not_equal(predicted_labels, true_labels))[0]

        # import pdb;pdb.set_trace()
        with open(f'misclassified_file_paths_epoch_{epoch}.txt', 'w') as file:
            for idx in missclass_indices:
                file.write(f"{self.validation_generator.filenames[idx]}\n")

def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.model_type == 'signal':
        model = SignalDetectionModel.build(intersection=config['intersection'],
                                           width=config['width'],
                                           height=config['height'],
                                           num_channels=config['num_channels'])
    elif args.model_type == 'train':
        model = TrainDetectionModel.build(width=config['width'],
                                          height=config['height'],
                                          num_channels=config['num_channels'])

        # model = MobileNetV3Small(
        #     input_shape=(config['width'], config['height'], config['num_channels']),
        #     alpha=1.0,
        #     include_top=True,
        #     weights=None,
        #     input_tensor=None,
        #     pooling=None,
        #     classes=2,
        #     classifier_activation="sigmoid"
        # )

        # # Load MobileNetV3Small without the top classification layer
        # base_model = MobileNetV3Small(
        #     input_shape=(config['width'], config['height'], config['num_channels']),
        #     include_top=False,
        #     weights='imagenet')

        # # Freeze the base model layers
        # for layer in base_model.layers:
        #     layer.trainable = False

        # # Add your own classification layers
        # x = base_model.output
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(128, activation='relu')(x)
        # x = Dense(64, activation='relu')(x)
        # output = Dense(1, activation='sigmoid')(x)  # Binary classification

        # # Create the final model
        # model = Model(inputs=base_model.input, outputs=output)
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
        CSVLogger(os.path.join(output_sub_dir, 'epochs.csv')),
        ConfusionMatrixCallback(validation_generator=validation_generator)
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
