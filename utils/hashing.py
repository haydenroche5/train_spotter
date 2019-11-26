import hashlib, os
import traceback


def get_dir_hash(directory, verbose=False):
    hash_value = hashlib.md5()
    if not os.path.exists(directory):
        raise Exception('Directory {} does not exist.'.format(directory))
    for root, dir_names, file_names in os.walk(directory):
        for file_name in file_names:
            path = os.path.join(root, file_name)
            if verbose:
                print('Hashing {}'.format(path))
            hash_value.update(path.encode())

    return hash_value.hexdigest()
