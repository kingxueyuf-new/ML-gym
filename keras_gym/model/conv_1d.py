from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

def build_model():
    seq_length = 64

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))