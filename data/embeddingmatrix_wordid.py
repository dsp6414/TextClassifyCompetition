import word2vec
import numpy as np


def precess_embedding(em_file, em_result):
    '''
    embedding ->numpy
    '''
    em = word2vec.load(em_file)
    vec = (em.vectors)
    word2id = em.vocab_hash
    np.savez_compressed(em_result, vector=vec, word2id=word2id)


if __name__ == '__main__':
    precess_embedding("../input/char.vec", "../input/char.npz")
    precess_embedding("../input/word.vec", "../input/word.npz")
