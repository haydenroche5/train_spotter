from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


class SignalDetectionModel:
    @staticmethod
    def build(width, height, num_channels):
        model = Sequential()
        model.add(
            Conv2D(8, (3, 3),
                   input_shape=(height, width, num_channels),
                   activation='relu',
                   padding='same'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))

        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        return model
