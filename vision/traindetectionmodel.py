from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense


class TrainDetectionModel:
    @staticmethod
    def build(width, height, num_channels):
        conv_kernel_size = (5, 5)

        model = Sequential()
        model.add(
            Conv2D(8,
                   conv_kernel_size,
                   input_shape=(height, width, num_channels),
                   activation='relu',
                   padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Dropout(0.3))

        model.add(
            Conv2D(16, conv_kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Dropout(0.3))

        model.add(
            Conv2D(32, conv_kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        return model
