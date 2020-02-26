import os
import argparse
import re


def main(args):
    hdf5_files = [
        os.path.join(args.model_dir, file)
        for file in os.listdir(args.model_dir) if file.endswith('.hdf5')
    ]
    num_models = len(hdf5_files)

    if args.n >= num_models:
        print(
            f'Nothing to do. N, {args.n}, >=    the number of model files, {num_models}.'
        )
        return

    for i, hdf5_file in enumerate(hdf5_files):
        hdf5_name = os.path.basename(hdf5_file)
        pattern = 'model\.\d{2}-(\d+\.\d+)\.hdf5'
        match = re.match(pattern, hdf5_name)

        if match:
            val_loss = float(match.group(1))
            hdf5_files[i] = (hdf5_files[i], val_loss)
        else:
            raise Exception(
                '{} doesn\'t match the expected pattern.'.format(hdf5_name))

    hdf5_files.sort(key=lambda x: x[1], reverse=True)
    num_models_to_del = num_models - args.n

    print(f'Will keep the top {args.n} models.')

    for i in range(num_models_to_del):
        print(f'Deleting {hdf5_files[i][0]}.')
        os.remove(hdf5_files[i][0])


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Cull model versions outside the top N.')
    arg_parser.add_argument(
        '-m',
        '--model-dir',
        dest='model_dir',
        required=True,
        help='The directory containing the HDF5 model files.')
    arg_parser.add_argument('-n',
                            dest='n',
                            default=1,
                            type=int,
                            help='The number of models to keep from the top.')
    main(arg_parser.parse_args())
