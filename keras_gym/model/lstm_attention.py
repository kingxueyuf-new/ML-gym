from keras.layers import LSTM


def build_model():
    from keras.layers.core import *
    from keras.models import Sequential

    input_dim = 32
    hidden = 32

    # The LSTM  model -  output_shape = (batch, step, hidden)
    model1 = Sequential()
    model1.add(LSTM(input_dim=input_dim, output_dim=hidden, input_length=step, return_sequences=True))

    # The weight model  - actual output shape  = (batch, step)
    # after reshape : output_shape = (batch, step,  hidden)
    model2 = Sequential()
    model2.add(Dense(input_dim=input_dim, output_dim=step))
    model2.add(Activation('softmax'))  # Learn a probability distribution over each  step.
    # Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
    model2.add(RepeatVector(hidden))
    model2.add(Permute(2, 1))

    # The final model which gives the weighted sum:
    model = Sequential()
    model.add(Merge([model1, model2], 'mul'))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
    model.add(TimeDistributedMerge('sum'))  # Sum the weighted elements.

    model.compile(loss='mse', optimizer='sgd')
