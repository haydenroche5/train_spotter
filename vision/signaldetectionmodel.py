from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


class SignalDetectionModel:
    @staticmethod
    def build(width, height, num_channels):
        max_pool_size = (2, 2)
        max_pool_stride = 2
        conv_kernel_size = (3, 3)

        model = Sequential()
        model.add(
            Conv2D(32, (5, 5),
                   input_shape=(height, width, num_channels),
                   activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=max_pool_stride))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, conv_kernel_size, activation='relu'))
        model.add(
            MaxPooling2D(pool_size=max_pool_size, strides=max_pool_stride))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, conv_kernel_size, activation='relu'))
        model.add(
            MaxPooling2D(pool_size=max_pool_size, strides=max_pool_stride))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        return model
