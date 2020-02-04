import os
import cv2

crop_upper_left_x = 470
crop_upper_left_y = 0
crop_width = 70
crop_height = 400
root_input_directory = '/home/cohub/datasets/train_chestnut/'
root_output_directory = '/home/cohub/datasets/signal_chestnut/'

input_subdirs = [
    os.path.join(root_input_directory, 'train'),
    os.path.join(root_input_directory, 'no_train')
]
output_subdirs = [
    os.path.join(root_output_directory, 'train'),
    os.path.join(root_output_directory, 'no_train')
]

imgs_done = 0
update_interval = 100
for input_dir, output_dir in zip(input_subdirs, output_subdirs):
    input_img_names = [i for i in os.listdir(input_dir) if i.endswith('.jpg')]
    for input_img_name in input_img_names:
        input_img_path = os.path.join(input_dir, input_img_name)
        output_img_path = os.path.join(output_dir, input_img_name)
        img = cv2.imread(input_img_path)
        cropped_img = img[crop_upper_left_y:crop_upper_left_y +
                          crop_height, crop_upper_left_x:crop_upper_left_x +
                          crop_width]
        cv2.imwrite(output_img_path, cropped_img)
        imgs_done += 1

        if (imgs_done % update_interval) == 0:
            print(f'Images done: {imgs_done}.', flush=True)

# TODO
# Create a config file that specifies:
# - output directory
# - input directory
# - upper left corner of crop
# - width
# - height