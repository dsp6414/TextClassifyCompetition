from gensim.models import Word2Vec
import gensim
import re, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def sentence2words(sentence, stopWords=False, stopWords_set=None):
    return sentence.split()

class MySentences(object):
    def __init__(self, list_csv):
        self.f = list_csv
    def __iter__(self):
        with open(self.f) as fn:
            for line in fn:
                if len(line) != 0:
                    yield sentence2words(line.strip())

    def train(self):
        num_features = 256
        min_word_count = 1
        num_workers = 24
        context = 5
        epoch = 40
        sample = 1e-5
        model = Word2Vec(
            self,
            size=num_features,
            min_count=min_word_count,
            workers=num_workers,
            sample=sample,
            window=context,
            iter=epoch,
        )
        return model

def train_save_word2vec(input_file, output_file):
    ms = MySentences(input_file)
    model = ms.train()
    model.wv.save_word2vec_format(output_file, binary=False)

if __name__ == "__main__":
    train_save_word2vec("../input/Char.txt", "../input/char.vec")
    train_save_word2vec("../input/word.txt", "../input/word.vec")