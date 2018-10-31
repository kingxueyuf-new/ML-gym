from keras.layers import *
from keras.models import *
from keras.engine.input_layer import Input
import keras.backend as K


def build_model(sequence_len=10, char_num=52, max_char_per_word=30):
    """

    :param sequence_len: sequence length
    :param char_num: how many char from a to z, A to Z
    :return:
    """

    word_input = Input((sequence_len, 300,))
    char_input = Input((sequence_len, max_char_per_word,))
    # x = (batch_size, sequence_len, max_char_per_word)

    x = TimeDistributed(Embedding(input_dim=char_num, output_dim=100, input_length=max_char_per_word))(char_input)
    # x = (batch_size, sequence_len, max_char_per_word, output_dim)
    print("---x shape---")
    print(x.shape)

    x = TimeDistributed(Conv1D(filters=300, kernel_size=3))(x)
    # x = (batch_size, sequence_len, new_steps, filters)
    ## new_steps = sequence_len-kernel_size+1
    ## filters = num of filters
    print("---x shape---")
    print(x.shape)

    x = TimeDistributed(MaxPooling1D(pool_size=max_char_per_word - 3 + 1))(x)
    # x = (batch_size, sequence_len, downsampled_steps, features)
    print("---x shape---")
    print(x.shape)

    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    # x = (batch_size, sequence_len, features)
    print("---x shape---")
    print(x.shape)

    x = Concatenate(axis=2)([x, word_input])
    # x = (batch_size, sequence_len, 600)
    x = Bidirectional(GRU(return_sequences=True))(x)

    output = Dense(units=1)(x)
    model = Model(inputs=[char_input, word_input], outputs=output)

    model.summary()
    return model


if __name__ == "__main__":
    build_model()
