import shutil
import glob
import os

data_dir                = '/home/hayden/proj/ml/train_detection/data/'
training_train_dir      = data_dir + 'training/train/'
training_no_train_dir   = data_dir + 'training/no_train/'
validation_train_dir    = data_dir + 'validation/train/'
validation_no_train_dir = data_dir + 'validation/no_train/'

extracted_train_dirs = ['/home/hayden/Downloads/Day/Present/']

for extracted_train_dir in extracted_train_dirs:
    jpg_files = glob.glob('{}*.jpg'.format(extracted_train_dir))
    jpg_count = len(jpg_files)
    training_count = int(0.9 * jpg_count)
    validation_count = jpg_count - training_count
    for filename in jpg_files:
        if training_count == 0:
            shutil.move(filename, validation_train_dir)
            validation_count -= 1
        else:
            shutil.move(filename, training_train_dir)
            training_count -= 1

    if training_count != 0 or validation_count != 0:
        print('Training count: {}'.format(training_count))
        print('Validation count: {}'.format(validation_count))

extracted_no_train_dirs = ['/home/hayden/Downloads/Day/Absent/']

for extracted_no_train_dir in extracted_no_train_dirs:
    jpg_files = glob.glob('{}*.jpg'.format(extracted_no_train_dir))
    jpg_count = len(jpg_files)
    training_count = int(0.9 * jpg_count)
    validation_count = jpg_count - training_count
    for filename in jpg_files:
        if training_count == 0:
            shutil.move(filename, validation_no_train_dir)
            validation_count -= 1
        else:
            shutil.move(filename, training_no_train_dir)
            training_count -= 1

    if training_count != 0 or validation_count != 0:
        print('Training count: {}'.format(training_count))
        print('Validation count: {}'.format(validation_count))
