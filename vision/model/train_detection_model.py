from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


class TrainDetectionModel:
    @staticmethod
    def build(width, height, num_channels):
        initial_max_pool_size = 3
        initial_max_pool_stride = 2
        latter_max_pool_size = 2
        latter_max_pool_stride = 2
        initial_conv_kernel_size = (11, 11)
        latter_conv_kernel_size = (3, 3)

        model = Sequential()
        model.add(
            Conv2D(16,
                   initial_conv_kernel_size,
                   input_shape=(height, width, num_channels),
                   activation='relu'))
        model.add(BatchNormalization())
        model.add(
            MaxPooling2D(pool_size=initial_max_pool_size,
                         strides=initial_max_pool_stride))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, latter_conv_kernel_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(
            MaxPooling2D(pool_size=latter_max_pool_size,
                         strides=latter_max_pool_stride))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, latter_conv_kernel_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(
            MaxPooling2D(pool_size=latter_max_pool_size,
                         strides=latter_max_pool_stride))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        return model