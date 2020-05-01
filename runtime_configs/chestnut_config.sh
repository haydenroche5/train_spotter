#!/bin/bash

usb_device="/dev/sda1"
usb_directory="/mnt/usb2"

root_directory="/home/pi/train_spotter/"
venv_file="${root_directory}/venv/bin/activate"
spotter_file="${root_directory}/spotter.py"
train_model_weights="${root_directory}/saved_models/chestnut/train/20200329_112403/model.29-0.0747.hdf5"
signal_model_weights="${root_directory}/saved_models/chestnut/signal/20200415_153017/model.09-0.0009.hdf5"
events_directory="${usb_directory}/events/"
logging_directory="${root_directory}/logging/"
stdout_stderr_file="${root_directory}/detector.log"

camera_ip="10.10.1.182"
threshold="0.90"
sleep_length="3"
