import gensim
import numpy as np


def w2v_export_npz(embedding_file):
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)
    embeddings = []
    vocabulary = model.vocab
    function_words = ['<PAD>', '<OOV>']
    word2id = dict((v, i) for i, v in enumerate(function_words))
    for i, word in enumerate(vocabulary, start=2):
        embeddings.append(model[word])
        word2id[word] = i
    print(f"we have {i} words")
    embeddings = np.array(embeddings, dtype=np.float32)

    weights = np.zeros((embeddings.shape[0] + 2, embeddings.shape[1]), np.float32)
    weights[1] = np.average(embeddings, axis=0)  # oov
    weights[2:] = embeddings
    np.savez_compressed(embedding_file+'.npz', wordid=word2id, weights=weights)
    print(f"file saved in {embedding_file}.npz !!")



if __name__ == '__main__':
    w2v_export_npz("../input/char.vec")
    w2v_export_npz("../input/word.vec")
