from keras.layers import *
from keras.models import *
from keras.engine.input_layer import Input
import keras.backend as K

from keras_gym.generator.generator import MyGenerator
from keras_gym.embedding.load_word2vec import load_word_embedding_word2vec_format
from keras_gym.preprocessing import read_csv
from keras_gym.preprocessing.char2idx import init_char_2_idx


def build_model(char_num=52, max_char_per_word=30):
    """

    :param sequence_len: sequence length
    :param char_num: how many char from a to z, A to Z
    :return:
    """

    word_input = Input((None, 300,))
    char_input = Input((None, max_char_per_word,))
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

    x = Bidirectional(GRU(units=300))(x)
    # x = (batch_size, sequence_len, 300+300)

    output = Dense(units=11, activation='softmax')(x)

    # x = (batch_size, sequence_len, 11)

    model = Model(inputs=[char_input, word_input], outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    return model


if __name__ == "__main__":
    char_num = 94
    max_char_per_word = 30
    num_classes = 10 + 1
    model = build_model(char_num=char_num + 1, max_char_per_word=max_char_per_word)
    word_embedding = load_word_embedding_word2vec_format("../../gensim_glove_vectorstwitter.27B.50d.txt")
    query_to_class_idx = read_csv("../../LSTM classifier label - Sheet1.csv")
    my_generator = MyGenerator(word_embedding,
                               max_char_per_word,
                               init_char_2_idx(),
                               num_classes,
                               query_to_class_idx.keys(),
                               query_to_class_idx.values())
    model.fit_generator(generator=my_generator,
                        workers=3,
                        shuffle=True)
