#!/bin/bash

# Read configuration variables based on IP address
ip_addr=$(hostname -I)
chestnut_ip_addr="10.10.1.191"
fourth_ip_addr="10.10.1.190"

if [ "$chestnut_ip_addr" == "$ip_addr" ]; then
    source /home/pi/train_detction/chestnut_config.sh
elif [ "$chestnut_ip_addr" == "$ip_addr" ]; then
    source /home/pi/train_detction/fourth_config.sh
else
    echo "Unrecognized IP address."
    exit 1
fi

# Mount the USB drive
mount $usb_device $usb_directory

# Check if the spotter script is running
if pgrep spotter > /dev/null 2>&1; then
    echo "Train spotter is already running."
else
    # TODO: remove --test
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
    -g $logging_directory > $stdout_stderr_file 2>&1 &"
fi

exit 0
