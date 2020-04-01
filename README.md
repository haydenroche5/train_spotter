# train_spotter
App for detecting trains in the Wedgewood-Houston neighborhood of Nashville, TN.

## Overview
This project uses a CNN-based image classification model to predict if a train is currently blocking the intersections of the railroad tracks at Chestnut St. and 4th Ave.

The guts of the code live in the `core` module. `detector.py` is responsible for grabbing images from webcams pointed at each intersection and running the images through the model. Predictions, formatted as a probability of train presence, [0.0, 1.0], are published over IPC using ZeroMQ. `webpublisher.py` subscribes to these updates and sends them along to a web server. That web server is responsible for updating the user-facing app. `eventtracker.py` saves images when a train is present along with some metadata. These three components run in separate processes and are spawned from `spotter.py`.

## Blog
You can read more about this project [here](https://cohub.com/train-spotter).
