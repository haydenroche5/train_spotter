import subprocess
from subprocess import CalledProcessError
import shlex
import argparse
import os.path


def main(args):
    if args.vpn:
        command = "nmcli con up id '{}'".format(args.vpn)
        tries = 5
        tries_remaining = tries
        while tries_remaining != 0:
            tries_remaining -= 1
            result = subprocess.run(shlex.split(command), capture_output=True)
            if result.returncode != 0:
                if 'is already active' in result.stderr.decode():
                    print('Already connected to VPN \'{}\'.'.format(args.vpn))
                    break
                print(
                    'Failed to connect to VPN \'{vpn}\'. Trying {tries_remaining} more time(s).'
                    .format(vpn=args.vpn, tries_remaining=tries_remaining))
                if tries == 0:
                    raise Exception(
                        'Couldn\'t connect to VPN \'{vpn}\' after {tries} attempts.'
                        .format(vpn=args.vpn, tries=tries))
            else:
                print('Connected to VPN {}.'.format(args.vpn))
                break

    model_dir = os.path.join(args.output_dir, 'saved_model')
    current_version_file = os.path.join(args.output_dir, 'current_version.txt')

    command = "scp -r {current_version_file} {detector_script} {model_dir} {user}@{ip}:{dest_dir}".format(
        current_version_file=current_version_file,
        detector_script=args.detector_script,
        model_dir=model_dir,
        user=args.dest_user,
        ip=args.dest_ip,
        dest_dir=args.dest_dir)
    subprocess.run(shlex.split(command), check=True)


# cat ~/.ssh/id_rsa.pub | ssh pi@10.10.1.190 'cat >> .ssh/authorized_keys'

# TODO: Take path to saved model, user, IP, destination path on target machine as args.

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Deploy the train detector.')
    arg_parser.add_argument(
        '--output-dir',
        dest='output_dir',
        required=True,
        help='Directory containing the model directory and version file.')
    arg_parser.add_argument('--detector-script',
                            dest='detector_script',
                            required=True,
                            help='Path to the train detector script.')
    arg_parser.add_argument('--dest-ip',
                            dest='dest_ip',
                            required=True,
                            help='IP address of the destination machine.')
    arg_parser.add_argument('--dest-user',
                            dest='dest_user',
                            required=True,
                            help='User on the destination machine.')
    arg_parser.add_argument(
        '--dest-dir',
        dest='dest_dir',
        required=True,
        help='Directory on the destination machine to copy the model to.')
    arg_parser.add_argument(
        '--vpn',
        dest='vpn',
        help=
        'Name of VPN to connect to, if deploying machine and destinatinon machine aren\'t on the same network.'
    )
    main(arg_parser.parse_args())