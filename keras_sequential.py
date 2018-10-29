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


def train(x_train, y_train):
    model.fit(x_train, y_train, epochs=5, batch_size=32)


def evaluate(x_eval, y_eval):
    loss_and_metrics = model.evaluate(x_eval, y_eval, batch_size=128)


def predict(x_test):
    classes = model.predict(x_test, batch_size=128)


if __name__ == "__main__":
    model = build_model()
