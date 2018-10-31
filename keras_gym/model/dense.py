import sys

print(sys.path)

from keras.models import Sequential
from keras.layers import Dense
import keras


def build_model():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam())

    return model


if __name__ == "__main__":
    model = build_model()
