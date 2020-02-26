import os
import argparse
import re
import shutil


def main(args):
    hdf5_files = [
        os.path.join(args.model_dir, file)
        for file in os.listdir(args.model_dir) if file.endswith('.hdf5')
    ]
    hdf5_names = [os.path.basename(f) for f in hdf5_files]

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

    hdf5_files_sorted = sorted(hdf5_files, key=lambda tup: tup[1])
    for f in hdf5_files_sorted:
        print(f)


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
