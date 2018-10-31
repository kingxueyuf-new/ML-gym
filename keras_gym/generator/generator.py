from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class MyGenerator(Sequence):

    def __init__(self, word_embedding, max_char_per_word, char2idx, num_classes, query, label, batch_size):
        self.word_embedding = word_embedding
        self.max_char_per_word = max_char_per_word
        self.char2idx = char2idx
        self.num_classes = num_classes
        self.query = query
        print(self.query)
        self.label = label
        print(self.label)
        self.batch_size = batch_size

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.

        ceil(109 / 10) = 11, so return 11
        """
        return np.ceil(len(self.query) / len(self.batch_size))

    def __getitem__(self, idx):
        """Gets batch at position `index`.

            # Arguments
                index: position of the batch in the Sequence.

            # Returns
                A batch

            ceil(109 / 10) = 11, so idx range from [0,10]
        """
        # Process input x
        query = self.query[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_sequence_word_embedding = []  # batch_size, sequence_len, 300
        batch_sequence_word_char_idx = []  # batch_size, sequence_len, char_len
        for sentence in query:
            words = Tokenizer(sentence)
            sentence_embedding = []
            sentence_word_char_idx = []
            for word in words:
                sentence_embedding.append(self.word_embedding[word])
                word_char_idx = []
                for char in word:
                    word_char_idx.append(self.char2idx[char])
                sentence_word_char_idx.append(word_char_idx)
            sentence_word_char_idx = pad_sequences(sentence_word_char_idx,
                                                   maxlen=self.max_char_per_word)  # pad word_char_idx to max_char_per_word
            batch_sequence_word_embedding.append(sentence_embedding)
            batch_sequence_word_char_idx.append(sentence_word_char_idx)

        # Process input y
        batch_y_labels = self.label[idx * self.batch_size: (idx + 1) * self.batch_size]  # batch_size

        # Convert label to one hot
        batch_y_labels = keras.utils.to_categorical(batch_y_labels, num_classes=self.num_classes)

        return [np.array(batch_sequence_word_char_idx), np.array(batch_sequence_word_embedding)], np.array(
            batch_y_labels)
