import argparse
import os
import re
import cv2
import pickle
import shutil


def verify_args(args):
    # Check the events directory
    if os.path.exists(args.events_dir) and os.path.isdir(args.events_dir):
        for events_dir_item in os.listdir(args.events_dir):
            events_subdir = os.path.join(args.events_dir, events_dir_item)

            if not os.path.isdir(events_subdir):
                raise Exception(
                    'Events directory item {} isn\'t a directory.'.format(
                        events_dir_item))

            # pattern = '\d+'
            # match = re.match(pattern, events_dir_item)

            # if not match:
            #     raise Exception(
            #         'Events subdirectoy {} doesn\'t match the expected pattern.'
            #         .format(events_dir_item))

            allowed_items = ['moments.pickle', 'images']
            for events_subdir_item in os.listdir(events_subdir):
                if events_subdir_item == 'moments.pickle':
                    pass
                elif events_subdir_item == 'images' and os.path.isdir(
                        os.path.join(events_subdir, events_subdir_item)):
                    pass
                else:
                    raise Exception(
                        'Events subdirectory {} contains unexpected item: {}.'.
                        format(events_subdir, events_subdir_item))

    else:
        raise Exception(
            'Events directory {} either doesn\'t exist or isn\'t a directory.'.
            format(args.events_dir))

    # Check the output directory
    if os.path.exists(args.output_dir) and os.path.isdir(args.output_dir):
        output_items = os.listdir(args.output_dir)
        num_items = len(output_items)

        if num_items != 2:
            raise Exception(
                'Expected output directory {} to contain two items. Contains {}.'
                .format(args.output_dir, num_items))

        allowed_subdirs = ['train', 'signal']
        for item in output_items:
            if not os.path.isdir(os.path.join(args.output_dir, item)):
                raise Exception(
                    'Output directory item {} isn\'t a directory.'.format(
                        item))
            elif item not in allowed_subdirs:
                raise Exception(
                    'Output directory subdirectories must be named train or signal. Subdirectory is named {}.'
                    .format(item))
    else:
        raise Exception(
            'Output directory {} either doesn\'t exist or isn\'t a directory.'.
            format(args.events_dir))

    # Check the threshold
    if args.threshold < 0 or args.threshold >= 1.0:
        raise Exception(
            'Threshold must be between 0 and 1.0. Threshold is {}.'.format(
                args.threshold))


def main(args):
    verify_args(args)

    events_subdirs = [
        os.path.join(args.events_dir, subdir)
        for subdir in os.listdir(args.events_dir)
    ]
    allowed_keys = [ord('n'), ord('t'), ord('s'), ord('q'), ord('f')]

    for events_subdir in events_subdirs:
        moments_file_path = os.path.join(events_subdir, 'moments.pickle')

        with open(moments_file_path, 'rb') as moments_file:
            moments = pickle.load(moments_file)

        for moment in moments:
            for subject in ['train', 'signal']:
                if subject == 'train':
                    prediction_value = moment['train_prediction_value']
                    img_path = moment['train_img_path']
                else:
                    prediction_value = moment['signal_prediction_value']
                    img_path = moment['signal_img_paths'][0]

                if prediction_value >= args.threshold:
                    base_img_name = os.path.basename(img_path)
                    img_file_path = os.path.join(events_subdir, 'images',
                                                 base_img_name)
                    new_img_file = str(
                        moment['timestamp']) + '_' + base_img_name
                    positive_img_file = os.path.join(args.output_dir, subject,
                                                     new_img_file)
                    negative_img_file = os.path.join(args.output_dir, subject,
                                                     'no_{}'.format(subject),
                                                     new_img_file)

                    if os.path.exists(positive_img_file) or os.path.exists(
                            negative_img_file):
                        print(f'{new_img_file} already exists, skipping.')
                        continue

                    if not os.path.exists(img_file_path):
                        print(f'{img_file_path} doesn\'t exist, skipping.')
                        continue

                    img = cv2.imread(img_file_path)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottom_left_corner_of_text = (10, img.shape[0] - 10)
                    font_scale = 1
                    font_color = (255, 0, 0)
                    font_thickness = 3
                    cv2.putText(img, '{:.5f}'.format(prediction_value),
                                bottom_left_corner_of_text, font, font_scale,
                                font_color, font_thickness)

                    cv2.imshow(img_file_path, img)
                    print(f'Displaying {img_file_path}.')
                    key = cv2.waitKey(0)

                    while key not in allowed_keys:
                        key = cv2.waitKey(0)
                        print(
                            'Invalid key. Valid keys: {}'.format(allowed_keys))

                    if key == ord('n'):
                        print('Adding {} to the no_{} folder.'.format(
                            img_file_path, subject))
                        shutil.copy(img_file_path, negative_img_file)
                    elif key == ord('t'):
                        print('Adding {} to the {} folder.'.format(
                            img_file_path, subject))
                        shutil.copy(img_file_path, positive_img_file)
                    elif key == ord('q'):
                        print('Quitting.')
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('s'):
                        print(f'Skipping event.')
                        cv2.destroyAllWindows()
                        break
                    elif key == ord('f'):
                        print(f'Skipping moment.')
                    cv2.destroyAllWindows()

        print('Done.')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Organize moments from an events directory.')
    arg_parser.add_argument(
        '-e',
        '--events-dir',
        dest='events_dir',
        required=True,
        help=
        'The directory containing event subdirectories. Each event subdirectory should have a moments.pickle file and an images/ directory.'
    )
    arg_parser.add_argument(
        '-t',
        '--threshold',
        dest='threshold',
        type=float,
        required=True,
        help='Train probability threshold for displaying a moment.')
    arg_parser.add_argument(
        '-o',
        '--output-dir',
        dest='output_dir',
        required=True,
        help=
        'The output directory. Should contain two subdirectories: train and no_train.'
    )
    main(arg_parser.parse_args())
