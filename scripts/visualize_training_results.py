import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    with open(args.history_file, 'rb') as hf:
        history = pickle.load(hf)
    num_epochs = len(history['loss'])
    epochs_array = np.arange(num_epochs)

    fig = plt.figure()
    loss_plot = fig.add_subplot(211, xlabel='Epochs', ylabel='Loss')
    val_loss_line = loss_plot.plot(epochs_array,
                                   history['val_loss'],
                                   label='Validation Loss',
                                   linewidth=2.0,
                                   marker='+',
                                   markersize=10.0)
    training_loss_line = loss_plot.plot(epochs_array,
                                        history['loss'],
                                        label='Training Loss',
                                        linewidth=2.0,
                                        marker='4',
                                        markersize=10.0)
    loss_plot.grid(True)
    loss_plot.legend()

    accuracy_plot = fig.add_subplot(212, xlabel='Epochs', ylabel='Accuracy')
    val_accuracy_line = accuracy_plot.plot(epochs_array,
                                           history['val_accuracy'],
                                           label='Validation Accuracy',
                                           linewidth=2.0,
                                           marker='+',
                                           markersize=10.0)
    accuracy_line = accuracy_plot.plot(epochs_array,
                                       history['accuracy'],
                                       label='Training Accuracy',
                                       linewidth=2.0,
                                       marker='4',
                                       markersize=10.0)
    accuracy_plot.grid(True)
    accuracy_plot.legend()

    plt.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Analyze training results from a history file.')
    arg_parser.add_argument(
        '--history-file',
        dest='history_file',
        required=True,
        help='Pickle file containing the training history.')
    main(arg_parser.parse_args())
