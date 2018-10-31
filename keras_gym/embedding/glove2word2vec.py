from gensim.scripts.glove2word2vec import glove2word2vec


def run(in_path, out_path):
    glove2word2vec(glove_input_file=in_path, word2vec_output_file=out_path)


if __name__ == "__main__":
    run(in_path="/Users/robinxue/Documents/ML-gym/glove.twitter.27B/glove.twitter.27B.50d.txt",
        out_path="../../gensim_glove_vectorstwitter.27B.50d.txt")
