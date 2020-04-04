#!/bin/bash

usb_device="/dev/sda1"
usb_directory="/mnt/usb2/"

root_directory="/home/pi/train_detection/"
venv_file="${root_directory}/venv/bin/activate"
spotter_file="${root_directory}/spotter.py"
train_model_weights="${root_directory}/saved_models/chestnut/train/model.27-0.0807.hdf5"
signal_model_weights="${root_directory}/saved_models/chestnut/signal/model.04-0.0010.hdf5"
events_directory="${usb_directory}/events/"
logging_directory="${root_directory}/logging/"
stdout_stderr_file="${root_directory}/detector.log"

intersection="chestnut"
camera_ip="10.10.1.182"
threshold="0.95"
sleep_length="3"
