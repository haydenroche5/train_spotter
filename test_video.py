import cv2
import argparse


def prepare_img_for_train_detection(img):
    image_array = np.array(img)
    image_array_scaled = image_array / 255.0
    image_array_scaled_expanded = np.expand_dims(image_array_scaled, axis=0)

    return image_array_scaled_expanded


def prepare_signal_img_chestnut(model, img):
    model_img_height, model_img_width = model.layers[0].input_shape[1:3]
    image_array = np.array(img)
    img_width = image_array.shape[1]
    signal_x = int((315 / 1920.0) * img_width)
    signal_y = 0
    cropped_img = image_array[signal_y:signal_y +
                              model_img_height, signal_x:signal_x +
                              model_img_width]
    cropped_img_scaled = cropped_img / 255.0
    cropped_img_scaled_expanded = np.expand_dims(cropped_img_scaled, axis=0)

    return cropped_img_scaled_expanded


def prepare_signal_imgs_fourth(model, img):
    signal_xs = [1090, 1218]
    signal_ys = [306, 515]

    input_img_height, input_img_width = model.layers[0].input_shape[1:3]
    cropped_imgs = []
    for x, y in zip(signal_xs, signal_ys):
        image_array = np.array(img)
        cropped_img = image_array[y:y + input_img_height, x:x +
                                  input_img_width]
        cropped_img_scaled = cropped_img / 255.0
        cropped_img_scaled_expanded = np.expand_dims(cropped_img_scaled,
                                                     axis=0)
        cropped_imgs.append(cropped_img_scaled_expanded)

    return cropped_imgs


def main(args):
    train_img_width = 1920
    train_img_height = 1080
    train_input_height = 384
    train_input_width = 216
    num_channels = 3

    if args.intersection == 'fourth':
        signal_input_height = 130
        signal_input_width = 130
    elif args.intersection == 'chestnut':
        signal_input_height = 180
        signal_input_width = 170
    else:
        raise Exception('Unrecognized intersection: {}.'.format(
            args.intersection))

    train_detection_model = TrainDetectionModel.build(
        width=train_input_width,
        height=train_input_height,
        num_channels=num_channels)
    signal_detection_model = SignalDetectionModel.build(
        width=signal_input_width,
        height=signal_input_height,
        num_channels=num_channels)
    train_detection_model.load_weights(args.train_model_weights)
    signal_detection_model.load_weights(args.signal_model_weights)

    vc = cv2.VideoCapture(args.video)
    success, img = vc.read()
    frame_idx = 0
    while success:
        print(f'Frame index: {frame_idx}.')

        if frame_idx % args.stride == 0:
            print('Running detector...')
            img_resized = cv2.resize(img,
                                     (train_input_width, train_input_height),
                                     interpolation=cv2.INTER_AREA)
            train_detection_input_img = prepare_img_for_train_detection(
                img_resized)

            if args.intersection == 'fourth':
                signal_detection_input_imgs = prepare_signal_imgs_fourth(
                    signal_detection_model, img)
                signal_prediction_values = []

                for signal_img in signal_detection_input_imgs:
                    signal_prediction_value = np.array(
                        signal_detection_model.predict_on_batch(
                            signal_img)).flatten()[0]
                    signal_prediction_values.append(signal_prediction_value)

                    print(
                        f'Signal prediction value: {signal_prediction_value}.')

                signal_prediction_value = max(signal_prediction_values).astype(
                    float)
            elif args.intersection == 'chestnut':
                signal_detection_input_img = prepare_signal_img_chestnut(
                    signal_detection_model, img)
                signal_prediction_value = np.array(
                    signal_detection_model.predict_on_batch(
                        signal_detection_input_img)).flatten()[0]
        else:
            print('Skipping...')

        train_prediction_value = np.array(
            train_detection_model.predict_on_batch(
                train_detection_input_img)).flatten()[0]
        print(f'Train prediction value: {train_prediction_value}.')
        print('------------------------------', flush=True)

        success, img = vc.read()
        frame_idx += 1


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Run the train detector on a recorded video.')
    arg_parser.add_argument('-v',
                            '--video',
                            dest='video',
                            required=True,
                            help='Path to the video file.')
    arg_parser.add_argument(
        '-i',
        '--intersection',
        dest='intersection',
        required=True,
        help=
        'The intersection that the camera is pointed at. One of \'chestnut\' or \'fourth\'.'
    )
    arg_parser.add_argument('-t',
                            '--train-model-weights',
                            dest='train_model_weights',
                            required=True,
                            help='Path to the train detection model weights.')
    arg_parser.add_argument('-s',
                            '--signal-model-weights',
                            dest='signal_model_weights',
                            required=True,
                            help='Path to the signal detection model weights.')
    arg_parser.add_argument(
        '-s',
        '--stride',
        dest='stride',
        required=True,
        help='The number of frames to skip between samples.')
    main(arg_parser.parse_args())