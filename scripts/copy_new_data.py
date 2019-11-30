import shutil
import glob
import os

data_dir = '/home/cohub/train_imgs/'
training_train_dir = data_dir + 'training/train/'
training_no_train_dir = data_dir + 'training/no_train/'
validation_train_dir = data_dir + 'validation/train/'
validation_no_train_dir = data_dir + 'validation/no_train/'

extracted_train_dirs = [
    '/home/cohub/test_src/Extracts/1106/Day/Present/',
    '/home/cohub/test_src/Extracts/1106/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1106/Night/Present/',
    '/home/cohub/test_src/Extracts/1106/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1108/Day/Present/',
    '/home/cohub/test_src/Extracts/1108/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1108/Night/Present/',
    '/home/cohub/test_src/Extracts/1108/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1112/Day/Present/',
    '/home/cohub/test_src/Extracts/1112/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1112/Night/Present/',
    '/home/cohub/test_src/Extracts/1112/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1114/Day/Present/',
    '/home/cohub/test_src/Extracts/1114/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1114/Night/Present/',
    '/home/cohub/test_src/Extracts/1114/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1116/Day/Present/',
    '/home/cohub/test_src/Extracts/1116/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1116/Night/Present/',
    '/home/cohub/test_src/Extracts/1116/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1118/Day/Present/',
    '/home/cohub/test_src/Extracts/1118/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1118/Night/Present/',
    '/home/cohub/test_src/Extracts/1118/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1120/Day/Present/',
    '/home/cohub/test_src/Extracts/1120/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1120/Night/Present/',
    '/home/cohub/test_src/Extracts/1120/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1122/Day/Present/',
    '/home/cohub/test_src/Extracts/1122/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1122/Night/Present/',
    '/home/cohub/test_src/Extracts/1122/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1124/Day/Present/',
    '/home/cohub/test_src/Extracts/1124/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1124/Night/Present/',
    '/home/cohub/test_src/Extracts/1124/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1107/Day/Present/',
    '/home/cohub/test_src/Extracts/1107/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1107/Night/Present/',
    '/home/cohub/test_src/Extracts/1107/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1111/Day/Present/',
    '/home/cohub/test_src/Extracts/1111/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1111/Night/Present/',
    '/home/cohub/test_src/Extracts/1111/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1113/Day/Present/',
    '/home/cohub/test_src/Extracts/1113/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1113/Night/Present/',
    '/home/cohub/test_src/Extracts/1113/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1115/Day/Present/',
    '/home/cohub/test_src/Extracts/1115/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1115/Night/Present/',
    '/home/cohub/test_src/Extracts/1115/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1117/Day/Present/',
    '/home/cohub/test_src/Extracts/1117/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1117/Night/Present/',
    '/home/cohub/test_src/Extracts/1117/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1119/Day/Present/',
    '/home/cohub/test_src/Extracts/1119/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1119/Night/Present/',
    '/home/cohub/test_src/Extracts/1119/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1121/Day/Present/',
    '/home/cohub/test_src/Extracts/1121/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1121/Night/Present/',
    '/home/cohub/test_src/Extracts/1121/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1123/Day/Present/',
    '/home/cohub/test_src/Extracts/1123/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1123/Night/Present/',
    '/home/cohub/test_src/Extracts/1123/Night/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1125/Day/Present/',
    '/home/cohub/test_src/Extracts/1125/Day/Present_not_blocking/',
    '/home/cohub/test_src/Extracts/1125/Night/Present/',
    '/home/cohub/test_src/Extracts/1125/Night/Present_not_blocking/'
]

for extracted_train_dir in extracted_train_dirs:
    jpg_files = glob.glob('{}*.jpg'.format(extracted_train_dir))
    jpg_count = len(jpg_files)
    training_count = int(0.9 * jpg_count)
    validation_count = jpg_count - training_count
    for filename in jpg_files:
        if training_count == 0:
            shutil.copy(filename, validation_train_dir)
            validation_count -= 1
        else:
            shutil.copy(filename, training_train_dir)
            training_count -= 1

    if training_count != 0 or validation_count != 0:
        print('Training count: {}'.format(training_count))
        print('Validation count: {}'.format(validation_count))

extracted_no_train_dirs = [
    '/home/cohub/test_src/Extracts/1106/Day/Absent/',
    '/home/cohub/test_src/Extracts/1106/Day/Close/',
    '/home/cohub/test_src/Extracts/1106/Night/Absent/',
    '/home/cohub/test_src/Extracts/1106/Night/Close/',
    '/home/cohub/test_src/Extracts/1108/Day/Absent/',
    '/home/cohub/test_src/Extracts/1108/Day/Close/',
    '/home/cohub/test_src/Extracts/1108/Night/Absent/',
    '/home/cohub/test_src/Extracts/1108/Night/Close/',
    '/home/cohub/test_src/Extracts/1112/Day/Absent/',
    '/home/cohub/test_src/Extracts/1112/Day/Close/',
    '/home/cohub/test_src/Extracts/1112/Night/Absent/',
    '/home/cohub/test_src/Extracts/1112/Night/Close/',
    '/home/cohub/test_src/Extracts/1114/Day/Absent/',
    '/home/cohub/test_src/Extracts/1114/Day/Close/',
    '/home/cohub/test_src/Extracts/1114/Night/Absent/',
    '/home/cohub/test_src/Extracts/1114/Night/Close/',
    '/home/cohub/test_src/Extracts/1116/Day/Absent/',
    '/home/cohub/test_src/Extracts/1116/Day/Close/',
    '/home/cohub/test_src/Extracts/1116/Night/Absent/',
    '/home/cohub/test_src/Extracts/1116/Night/Close/',
    '/home/cohub/test_src/Extracts/1118/Day/Absent/',
    '/home/cohub/test_src/Extracts/1118/Day/Close/',
    '/home/cohub/test_src/Extracts/1118/Night/Absent/',
    '/home/cohub/test_src/Extracts/1118/Night/Close/',
    '/home/cohub/test_src/Extracts/1120/Day/Absent/',
    '/home/cohub/test_src/Extracts/1120/Day/Close/',
    '/home/cohub/test_src/Extracts/1120/Night/Absent/',
    '/home/cohub/test_src/Extracts/1120/Night/Close/',
    '/home/cohub/test_src/Extracts/1122/Day/Absent/',
    '/home/cohub/test_src/Extracts/1122/Day/Close/',
    '/home/cohub/test_src/Extracts/1122/Night/Absent/',
    '/home/cohub/test_src/Extracts/1122/Night/Close/',
    '/home/cohub/test_src/Extracts/1124/Day/Absent/',
    '/home/cohub/test_src/Extracts/1124/Day/Close/',
    '/home/cohub/test_src/Extracts/1124/Night/Absent/',
    '/home/cohub/test_src/Extracts/1124/Night/Close/',
    '/home/cohub/test_src/Extracts/1107/Day/Absent/',
    '/home/cohub/test_src/Extracts/1107/Day/Close/',
    '/home/cohub/test_src/Extracts/1107/Night/Absent/',
    '/home/cohub/test_src/Extracts/1107/Night/Close/',
    '/home/cohub/test_src/Extracts/1111/Day/Absent/',
    '/home/cohub/test_src/Extracts/1111/Day/Close/',
    '/home/cohub/test_src/Extracts/1111/Night/Absent/',
    '/home/cohub/test_src/Extracts/1111/Night/Close/',
    '/home/cohub/test_src/Extracts/1113/Day/Absent/',
    '/home/cohub/test_src/Extracts/1113/Day/Close/',
    '/home/cohub/test_src/Extracts/1113/Night/Absent/',
    '/home/cohub/test_src/Extracts/1113/Night/Close/',
    '/home/cohub/test_src/Extracts/1115/Day/Absent/',
    '/home/cohub/test_src/Extracts/1115/Day/Close/',
    '/home/cohub/test_src/Extracts/1115/Night/Absent/',
    '/home/cohub/test_src/Extracts/1115/Night/Close/',
    '/home/cohub/test_src/Extracts/1117/Day/Absent/',
    '/home/cohub/test_src/Extracts/1117/Day/Close/',
    '/home/cohub/test_src/Extracts/1117/Night/Absent/',
    '/home/cohub/test_src/Extracts/1117/Night/Close/',
    '/home/cohub/test_src/Extracts/1119/Day/Absent/',
    '/home/cohub/test_src/Extracts/1119/Day/Close/',
    '/home/cohub/test_src/Extracts/1119/Night/Absent/',
    '/home/cohub/test_src/Extracts/1119/Night/Close/',
    '/home/cohub/test_src/Extracts/1121/Day/Absent/',
    '/home/cohub/test_src/Extracts/1121/Day/Close/',
    '/home/cohub/test_src/Extracts/1121/Night/Absent/',
    '/home/cohub/test_src/Extracts/1121/Night/Close/',
    '/home/cohub/test_src/Extracts/1123/Day/Absent/',
    '/home/cohub/test_src/Extracts/1123/Day/Close/',
    '/home/cohub/test_src/Extracts/1123/Night/Absent/',
    '/home/cohub/test_src/Extracts/1123/Night/Close/',
    '/home/cohub/test_src/Extracts/1125/Day/Absent/',
    '/home/cohub/test_src/Extracts/1125/Day/Close/',
    '/home/cohub/test_src/Extracts/1125/Night/Absent/',
    '/home/cohub/test_src/Extracts/1125/Night/Close/'
]

for extracted_no_train_dir in extracted_no_train_dirs:
    jpg_files = glob.glob('{}*.jpg'.format(extracted_no_train_dir))
    jpg_count = len(jpg_files)
    training_count = int(0.9 * jpg_count)
    validation_count = jpg_count - training_count
    for filename in jpg_files:
        if training_count == 0:
            shutil.copy(filename, validation_no_train_dir)
            validation_count -= 1
        else:
            shutil.copy(filename, training_no_train_dir)
            training_count -= 1

    if training_count != 0 or validation_count != 0:
        print('Training count: {}'.format(training_count))
        print('Validation count: {}'.format(validation_count))
