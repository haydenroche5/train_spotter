#!/bin/bash

# Exit if any command below fails.
set -e

# Clear the startup.log file
startup_log=/home/pi/train_spotter/startup.log
> $startup_log

intersection=$(cat /home/pi/intersection | sed 's/ *$//')
api_secret=$(cat /home/pi/api_secret | sed 's/ *$//')

# Read configuration variables based on intersection file
if [ "chestnut" == "$intersection" ]; then
    source /home/pi/train_spotter/runtime_configs/chestnut_config.sh >> $startup_log 2>&1
elif [ "fourth" == "$intersection" ]; then
    source /home/pi/train_spotter/runtime_configs/fourth_config.sh >> $startup_log 2>&1
else
    echo "Unrecognized intersection: $intersection." >> $startup_log
    exit 1
fi

# Mount the USB drive
if ! grep -qs "$usb_directory " /proc/mounts; then
    mount $usb_device $usb_directory >> $startup_log 2>&1
fi

# Check if the spotter script is running
if pgrep -f "train_spotter.py" > /dev/null 2>&1; then
    echo "Train spotter is already running." >> $startup_log
else
    su -c "source $venv_file && \
    nohup python $spotter_file \
    -i $intersection \
    -t $train_model_weights \
    -s $signal_model_weights \
    -e $events_directory \
    -c $camera_ip \
    -r $threshold \
    -l $sleep_length \
    -g $logging_directory \
    --api-secret $api_secret \
    > $stdout_stderr_file 2>&1 &" - pi
fi

exit 0
