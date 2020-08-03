#!/bin/bash

usb_device="/dev/sda1"
usb_directory="/mnt/usb2"

root_directory="/home/pi/train_spotter/"
venv_file="${root_directory}/venv/bin/activate"
spotter_file="${root_directory}/spotter.py"
train_model_weights="${root_directory}/saved_models/chestnut/train/20200731_162802/model.20-0.0646.hdf5"
signal_model_weights="${root_directory}/saved_models/chestnut/signal/20200715_163234/model.04-0.0102.hdf5"
events_directory="${usb_directory}/events/"
logging_directory="${root_directory}/logging/"
stdout_stderr_file="${root_directory}/detector.log"

camera_ip="10.10.1.182"
threshold="0.90"
sleep_length="3"
contrast_alpha="2.0"
contrast_threshold="100"
