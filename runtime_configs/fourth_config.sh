#!/bin/bash

usb_device="/dev/sda1"
usb_directory="/mnt/usb"

root_directory="/home/pi/train_spotter/"
venv_file="${root_directory}/venv/bin/activate"
spotter_file="${root_directory}/spotter.py"
train_model_weights="${root_directory}/saved_models/fourth/train/20200329_165428/model.29-0.0538.hdf5"
signal_model_weights="${root_directory}/saved_models/fourth/signal/20200413_225922/model.02-0.0062.hdf5"
events_directory="${usb_directory}/events/"
logging_directory="${root_directory}/logging/"
stdout_stderr_file="${root_directory}/detector.log"

camera_ip="10.10.1.181"
threshold="0.90"
sleep_length="3"
