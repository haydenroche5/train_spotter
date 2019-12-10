import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import os
import os.path
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix


class GradCamHeatMapper:
    def __init__(self, ref_model_dir):
        ref_model = load_model(ref_model_dir)
        last_conv_layer_name = ''
        for layer in ref_model.layers:
            config = layer.get_config()
            if config['name'].startswith('conv2d'):
                last_conv_layer_name = config['name']

        self.grad_model = tf.keras.models.Model([ref_model.inputs], [
            ref_model.get_layer(last_conv_layer_name).output, ref_model.output
        ])
        self.grad_model.layers[-1].activation = None

    def get_heat_map(self, img):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(np.array([img]))
            score = predictions[:, 0]

        output = conv_outputs[0]
        grads = tape.gradient(score, conv_outputs)[0]

        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(
            grads > 0, 'float32') * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.ones(output.shape[0:2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        height, width = img.shape[0:2]
        cam = cv2.resize(cam.numpy(), (width, height))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        output_image_bgr = cv2.addWeighted(
            cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1,
            0)
        output_image_rgb = output_image_bgr[:, :, ::-1]

        return output_image_rgb


class HistoryRecorder(Callback):
    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def on_train_begin(self, logs={}):
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_accuracy.append(logs.get('val_accuracy'))

    def as_dict(self):
        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'val_loss': self.val_loss,
            'val_accuracy': self.val_accuracy
        }


class Model:
    def __init__(self, img_height, img_width, num_channels, data_dir_path,
                 batch_size):
        training_img_gen = ImageDataGenerator(rescale=1. / 255)
        validation_img_gen = ImageDataGenerator(rescale=1. / 255)
        self.training_path = os.path.join(data_dir_path, 'training')
        self.validation_path = os.path.join(data_dir_path, 'validation')
        self.training_generator = training_img_gen.flow_from_directory(
            self.training_path,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True)
        self.validation_generator = validation_img_gen.flow_from_directory(
            self.validation_path,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False)
        self.batch_size = batch_size
        self.history_recorder = HistoryRecorder()

        self.num_training_samples = sum(
            [len(files) for _, _, files in os.walk(self.training_path)])
        self.num_validation_samples = sum(
            [len(files) for _, _, files in os.walk(self.validation_path)])

        initial_max_pool_size = (3, 3)
        initial_max_pool_stride = 2
        latter_max_pool_size = (2, 2)
        latter_max_pool_stride = 2
        initial_conv_kernel_size = (11, 11)
        latter_conv_kernel_size = (3, 3)

        self.network = Sequential()
        self.network.add(
            Conv2D(32,
                   initial_conv_kernel_size,
                   input_shape=(img_height, img_width, num_channels),
                   activation='relu'))
        self.network.add(
            MaxPooling2D(pool_size=initial_max_pool_size,
                         strides=initial_max_pool_stride))
        self.network.add(Dropout(0.2))

        self.network.add(Conv2D(32, latter_conv_kernel_size,
                                activation='relu'))
        self.network.add(
            MaxPooling2D(pool_size=latter_max_pool_size,
                         strides=latter_max_pool_stride))
        self.network.add(Dropout(0.2))

        self.network.add(Conv2D(64, latter_conv_kernel_size,
                                activation='relu'))
        self.network.add(
            MaxPooling2D(pool_size=latter_max_pool_size,
                         strides=latter_max_pool_stride))
        self.network.add(Dropout(0.2))

        self.network.add(Flatten())
        self.network.add(Dense(64, activation='relu'))
        self.network.add(Dropout(0.5))
        self.network.add(Dense(1, activation='sigmoid'))

        self.network.compile(loss='binary_crossentropy',
                             optimizer=SGD(lr=1e-4, momentum=0.9),
                             metrics=['accuracy'])

        print(self.network.summary())

    def load_pretrained(self, saved_model_dir_path):
        self.network = load_model(saved_model_dir_path)

    def train(self, num_epochs, patience, output_dir_path, analyze=True):
        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=patience,
                          restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.join(output_dir_path,
                                                  'saved_model'),
                            monitor='val_loss',
                            save_best_only=True), self.history_recorder
        ]
        self.network.fit_generator(
            self.training_generator,
            steps_per_epoch=math.ceil(1.0 * self.num_training_samples /
                                      self.batch_size),
            epochs=num_epochs,
            callbacks=callbacks,
            validation_data=self.validation_generator,
            validation_steps=math.ceil(1.0 * self.num_validation_samples /
                                       self.batch_size))

        history = self.history_recorder.as_dict()
        history_file_path = os.path.join(output_dir_path, 'history.pickle')
        with open(history_file_path, 'wb') as history_file:
            pickle.dump(history, history_file)

    # TODO: distinguish between interactive and non-interactive mode
    def analyze_training_results(self, output_dir_path, interactive=False):
        history_file_path = os.path.join(output_dir_path, 'history.pickle')
        with open(history_file_path, 'rb') as history_file:
            history = pickle.load(history_file)
        num_epochs = len(history['loss'])
        epochs_array = np.arange(num_epochs)

        val_loss_line = plt.plot(epochs_array,
                                 history['val_loss'],
                                 label='Validation Loss')
        training_loss_line = plt.plot(epochs_array,
                                      history['loss'],
                                      label='Training Loss')
        plt.setp(val_loss_line, linewidth=2.0, marker='+', markersize=10.0)
        plt.setp(training_loss_line,
                 linewidth=2.0,
                 marker='4',
                 markersize=10.0)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()

        val_accuracy_line = plt.plot(epochs_array,
                                     history['val_accuracy'],
                                     label='Validation Accuracy')
        accuracy_line = plt.plot(epochs_array,
                                 history['accuracy'],
                                 label='Training Accuracy')
        plt.setp(val_accuracy_line, linewidth=2.0, marker='+', markersize=10.0)
        plt.setp(accuracy_line, linewidth=2.0, marker='4', markersize=10.0)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.show()

        true_labels = self.validation_generator.classes
        pred_labels = self.network.predict_generator(
            self.validation_generator).flatten()
        class_idx_dict = self.validation_generator.class_indices

        # Round to nearest class label (1 "train" or 0 "no_train")
        pred_class_idxs = np.around(pred_labels)
        filenames = self.validation_generator.filenames
        miss_count = 0
        grad_cam_heat_mapper = GradCamHeatMapper(
            os.path.join(output_dir_path, 'saved_model'))
        train_misses_path = os.path.join(output_dir_path, 'misses/train/')
        no_train_misses_path = os.path.join(output_dir_path,
                                            'misses/no_train/')
        if not os.path.exists(train_misses_path):
            os.makedirs(train_misses_path)
        if not os.path.exists(no_train_misses_path):
            os.makedirs(no_train_misses_path)

        for i in range(self.num_validation_samples):
            if int(pred_class_idxs[i]) != true_labels[i]:
                miss_count += 1
                batch_number = i // self.batch_size
                img_number = i % self.batch_size
                img = self.validation_generator[batch_number][0][img_number]
                img_heatmap = grad_cam_heat_mapper.get_heat_map(img)

                misses_path = ''
                if true_labels[i] == class_idx_dict['no_train']:
                    print(
                        'Model said {} contained a train ({}), but it didn\'t.'
                        .format(filenames[i], pred_labels[i]))
                    misses_path = no_train_misses_path
                else:
                    print(
                        'Model said {} did not contain a train ({}), but it did.'
                        .format(filenames[i], pred_labels[i]))
                    misses_path = train_misses_path

                if interactive:
                    plt.imshow(img)
                    plt.show()
                    plt.imshow(img_heatmap)
                    plt.show()

                # miss_img = cv2.hconcat([img, img_heatmap])
                # miss_file_path = os.path.join(misses_path,
                #                               '{}.jpg'.format(miss_count))
                # cv2.imwrite(miss_file_path, miss_img)

        print('Misclassified {} images.'.format(miss_count))

        cm = confusion_matrix(true_labels, pred_class_idxs)
        print('Confusion Matrix:')
        print(cm)
        true_negatives = cm[0][0]
        false_negatives = cm[1][0]
        true_positives = cm[1][1]
        false_positives = cm[0][1]
        print('True negatives: {}'.format(true_negatives))
        print('False negatives: {}'.format(false_negatives))
        print('True positives: {}'.format(true_positives))
        print('False positives: {}'.format(false_positives))
