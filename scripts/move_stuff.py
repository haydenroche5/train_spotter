import os
import shutil

# 1-89 train
# 90-98 remove
# 99-103 no_train

no_train_bound = 64
remove_bound = 114
train_bound = 128
root = '1123_nt12_'
num_imgs = len([file for file in os.listdir('.') if file.endswith('.jpg')])

for i in range(num_imgs):
    img_number = i + 1
    filename = root + f'{img_number:06}.jpg' 
    if img_number <= no_train_bound:
        shutil.move(filename, 'no_train/')
    elif img_number > no_train_bound and img_number <= remove_bound:
        pass
    elif img_number > remove_bound and img_number <= train_bound:
        shutil.move(filename, 'train/')
    else:
        raise Exception('Bounds are fucked up')
