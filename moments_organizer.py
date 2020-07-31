import argparse
import os
import re
import cv2
import pickle
import shutil
from datetime import datetime


def filter_events(args):
    events_subdirs = [
        os.path.join(args.events_dir, subdir)
        for subdir in os.listdir(args.events_dir)
    ]
    events_subdirs_filtered = []

    for events_subdir in events_subdirs:
        event_number = int(os.path.basename(events_subdir))
        if event_number < args.start_event:
            print(
                f'Skipping event #{event_number}, less than start event {args.start_event}.'
            )
            continue

        moments_file_path = os.path.join(events_subdir, 'moments.pickle')
        with open(moments_file_path, 'rb') as moments_file:
            moments = pickle.load(moments_file)

        if len(moments) < args.floor:
            print(f'Skipping event #{event_number}, too few moments.')
            continue

        if len(moments) > args.ceiling:
            print(f'Skipping event #{event_number}, too many moments.')
            continue

        events_subdirs_filtered.append(events_subdir)

    return events_subdirs_filtered


def main(args):
    events_subdirs = filter_events(args)

    allowed_keys = [ord('n'), ord('t'), ord('s'), ord('q'), ord('f')]

    subjects = []
    if args.show_train:
        subjects.append('train')
    if args.show_signal:
        subjects.append('signal')

    if len(subjects) == 0:
        raise Exception('Must specify one or both of --train, --signal.')

    for events_subdir in events_subdirs:
        moments_file_path = os.path.join(events_subdir, 'moments.pickle')

        with open(moments_file_path, 'rb') as moments_file:
            moments = pickle.load(moments_file)

        for moment in moments:
            for subject in subjects:
                if subject == 'train':
                    prediction_value = moment['train_prediction_value']
                    img_path = moment['train_img_path']
                else:
                    prediction_value = moment['signal_prediction_value']
                    img_path = moment['signal_img_paths'][0]

                base_img_name = os.path.basename(img_path)
                img_file_path = os.path.join(events_subdir, 'images',
                                             base_img_name)
                new_img_file = str(moment['timestamp']) + '_' + base_img_name
                positive_img_file = os.path.join(args.output_dir, subject,
                                                 new_img_file)
                negative_img_file = os.path.join(args.output_dir, subject,
                                                 f'no_{subject}', new_img_file)

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
                    print('Invalid key. Valid keys: {}'.format(allowed_keys))

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
    arg_parser.add_argument('--train',
                            dest='show_train',
                            default=False,
                            action='store_true',
                            help='Show images for the train model.')
    arg_parser.add_argument('--signal',
                            dest='show_signal',
                            default=False,
                            action='store_true',
                            help='Show images for the signal model.')
    arg_parser.add_argument(
        '--floor',
        dest='floor',
        type=int,
        default=0,
        help='# moments must exceed this value to be considered an "event".')
    arg_parser.add_argument(
        '--ceiling',
        dest='ceiling',
        type=int,
        default=10,
        help='# moments must be below this value to be considered an "event".')
    arg_parser.add_argument(
        '-o',
        '--output-dir',
        dest='output_dir',
        required=True,
        help=
        'The output directory. Should contain two subdirectories: train and no_train.'
    )
    arg_parser.add_argument(
        '--start-event',
        dest='start_event',
        type=int,
        required=True,
        help='Only consider events after and including this event number.')
    main(arg_parser.parse_args())
