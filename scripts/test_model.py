import os
import sys
from os.path import dirname, abspath
root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)
from vision.analysis import GradCamHeatMapper

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix


def main(args):
    height = 130
    width = 130
    num_channels = 3
    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 32

    img_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.9)
    testing_generator = img_gen.flow_from_directory(
        args.data_dir,  # same directory as training data
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False)

    num_testing_samples = len(testing_generator.labels)

    model = load_model(args.model_dir)
    grad_cam_heat_mapper = GradCamHeatMapper(args.model_dir)
    true_labels = testing_generator.classes
    pred_labels = model.predict_generator(testing_generator).flatten()
    class_idx_dict = testing_generator.class_indices

    pred_class_idxs = np.around(pred_labels)
    filenames = testing_generator.filenames
    miss_count = 0

    for i in range(num_testing_samples):
        if int(pred_class_idxs[i]) != true_labels[i]:
            miss_count += 1
            batch_number = i // batch_size
            img_number = i % batch_size
            img = testing_generator[batch_number][0][img_number]
            img_heatmap = grad_cam_heat_mapper.get_heat_map(img)

            if true_labels[i] == class_idx_dict['Absent']:
                print('Model said {} contained a signal ({}), but it didn\'t.'.
                      format(filenames[i], pred_labels[i]))
            else:
                print(
                    'Model said {} did not contain a signal ({}), but it did.'.
                    format(filenames[i], pred_labels[i]))

            plt.imshow(img)
            plt.show()
            plt.imshow(img_heatmap)
            plt.show()

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


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Test the signal detection model.')
    arg_parser.add_argument('--model-dir',
                            dest='model_dir',
                            required=True,
                            help='Directory of saved signal detection model.')
    arg_parser.add_argument('--data-dir',
                            dest='data_dir',
                            required=True,
                            help='Directory containing test data.')
    main(arg_parser.parse_args())
