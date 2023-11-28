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
    events_subdirs_sorted = sorted(events_subdirs, key=lambda x: int(os.path.basename(x)), reverse=args.reverse)
    events_subdirs_filtered = []

    for idx, events_subdir in enumerate(events_subdirs_sorted):
        if not (idx % args.event_stride == 0):
            continue

        event_number = int(os.path.basename(events_subdir))
        if (not args.reverse and event_number < args.start_event) or (args.reverse and event_number > args.start_event):
            print(
                f'Skipping event #{event_number}, less than start event {args.start_event}.'
            )
            continue

        moments_file_path = os.path.join(events_subdir, 'moments.pickle')
        if not os.path.exists(moments_file_path):
            print(f'Skipping event #{event_number}, no moments.pickle file.')
            continue

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
        next_event = False

        moments_file_path = os.path.join(events_subdir, 'moments.pickle')

        with open(moments_file_path, 'rb') as moments_file:
            moments = pickle.load(moments_file)

        for idx, moment in enumerate(moments):
            for subject in subjects:
                if subject == 'train':
                    if 'train_prediction_value' not in moment or 'train_img_path' not in moment:
                        print('Old moment format. Skipping event.')
                        break

                    prediction_value = moment['train_prediction_value']
                    img_path = moment['train_img_path']
                else:
                    prediction_value = moment['signal_prediction_value']
                    img_path = moment['signal_img_paths'][0]

                in_pred_bounds = prediction_value >= args.pred_lower and \
                                 prediction_value <= args.pred_upper
                if not in_pred_bounds:
                    continue
                if not (idx % args.image_stride == 0):
                    continue

                base_img_name = os.path.basename(img_path)
                img_file_path = os.path.join(events_subdir, 'images',
                                             base_img_name)
                new_img_file = str(moment['timestamp']) + '_' + base_img_name
                positive_img_file = os.path.join(args.output_dir, subject,
                                                 new_img_file)
                negative_img_file = os.path.join(args.output_dir, f'no_{subject}', new_img_file)

                if os.path.exists(positive_img_file) or os.path.exists(
                        negative_img_file):
                    print(f'{new_img_file} already exists, skipping.')
                    continue

                if not os.path.exists(img_file_path):
                    print(f'{img_file_path} doesn\'t exist, skipping.')
                    continue

                img = cv2.imread(img_file_path)
                if img is None:
                    print(f'Failed to read {img_file_path}, skipping.')
                    continue

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
                    print(f'New file: {negative_img_file}')
                    shutil.copy(img_file_path, negative_img_file)
                elif key == ord('t'):
                    print('Adding {} to the {} folder.'.format(
                        img_file_path, subject))
                    print(f'New file: {positive_img_file}')
                    shutil.copy(img_file_path, positive_img_file)
                elif key == ord('q'):
                    print('Quitting.')
                    cv2.destroyAllWindows()
                    return
                elif key == ord('f'):
                    print(f'Skipping event.')
                    next_event = True
                elif key == ord('s'):
                    print(f'Skipping moment.')

                cv2.destroyAllWindows()

                if next_event:
                    break

            if next_event:
                break

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
        type=float,
        default=float('inf'),
        help='# moments must be below this value to be considered an "event".')
    arg_parser.add_argument(
        '--pred-lower',
        dest='pred_lower',
        type=float,
        default=0,
        help='Prediction value lower bound.')
    arg_parser.add_argument(
        '--pred-upper',
        dest='pred_upper',
        type=float,
        default=1.0,
        help='Prediction value upper bound.')
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
        default=True,
        help='Only consider events after and including this event number.')
    arg_parser.add_argument(
        '--image-stride',
        dest='image_stride',
        type=int,
        default=1,
        help='Number of images to step over.')
    arg_parser.add_argument(
        '--event-stride',
        dest='event_stride',
        type=int,
        default=1,
        help='Number of events to step over.')
    arg_parser.add_argument(
        '--negative',
        dest='negative',
        default=False,
        action='store_true',
        help='Just look at negative images.')
    arg_parser.add_argument(
        '--reverse',
        dest='reverse',
        default=False,
        action='store_true',
        help='Process events in reverse event number order.')
    main(arg_parser.parse_args())
