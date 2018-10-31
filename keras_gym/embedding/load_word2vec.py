import gensim


def load_word_embedding_word2vec_format(path, is_binary=False):
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=is_binary)


if __name__ == "__main__":
    model = load_word_embedding_word2vec_format(
        "/Users/robinxue/Documents/ML-gym/gensim_glove_vectorstwitter.27B.50d.txt")
    print(model.vocab)
    print(model['dog'].shape)
