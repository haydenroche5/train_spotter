#!/bin/bash

usb_device="dev/sda1"
usb_directory="/mnt/usb/"

root_directory="/home/pi/train_detection/"
venv_file="${root_directory}/venv/bin/activate"
spotter_file="${root_directory}/spotter.py"
train_model_weights="${root_directory}/saved_models/fourth/train/model.31-0.0609.hdf5"
signal_model_weights="${root_directory}/saved_models/fourth/signal/model.08-0.0280.hdf5"
events_directory="${usb_directory}/events/"
logging_directory="${root_directory}/logging/"
stdout_stderr_file="${root_directory}/detector.log"

intersection="fourth"
camera_ip="10.10.1.181"
threshold="0.95"
sleep_length="3"
