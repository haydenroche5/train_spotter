#!/bin/bash

# Clear the old stdout/stderr file
> $stdout_stderr_file

intersection=$(cat /home/pi/intersection | sed 's/ *$//')
api_secret=$(cat /home/pi/api_secret | sed 's/ *$//')

# Read configuration variables based on intersection file
if [ "chestnut" == "$intersection" ]; then
    source /home/pi/train_spotter/runtime_configs/chestnut_config.sh >> $stdout_stderr_file 2>&1
elif [ "fourth" == "$intersection" ]; then
    source /home/pi/train_spotter/runtime_configs/fourth_config.sh >> $stdout_stderr_file 2>&1
else
    echo "Unrecognized intersection: $intersection." >> $stdout_stderr_file
    exit 1
fi

# Mount the USB drive
mount $usb_device $usb_directory >> $stdout_stderr_file 2>&1

# Check if the spotter script is running
if pgrep -f spotter > /dev/null 2>&1; then
    echo "Train spotter is already running." >> $stdout_stderr_file
else
    # TODO: remove --test when ready to go live
    su - pi -c "source $venv_file && \
    python $spotter_file \
    -i $intersection \
    --test \
    -t $train_model_weights \
    -s $signal_model_weights \
    -e $events_directory \
    -c $camera_ip \
    -r $threshold \
    -l $sleep_length \
    -g $logging_directory \
    --api-secret $api_secret \
    >> $stdout_stderr_file 2>&1 &"
fi

exit 0
