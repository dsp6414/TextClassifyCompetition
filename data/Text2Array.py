# encoding:utf-8
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import os
from os import path
from sklearn.model_selection import train_test_split

char2id = np.load('../input/char.vec.npz')['wordid'][()]
word2id = np.load('../input/word.vec.npz')['wordid'][()]
save_path = "../input/pt"
train_path = "../input/pt/train"
validation_path = "../input/pt/validation"
test_path = "../input/pt/validation"
for p in [save_path, train_path, validation_path, test_path]:
    if not path.exists(p):
        os.makedirs(p, exist_ok=True)


def char_col_to_id_list(df):
    articles = df['article'].apply(
        lambda x: [char2id.get(s, 1) for s in x.split()])
    padded_char_list = pad_sequences(articles.values, maxlen=4000,
                                     padding='post',
                                     truncating='post')
    return padded_char_list


def word_col_to_id_list(df):
    words = df['article'].apply(
        lambda x: [word2id.get(s, 1) for s in x.split()])
    padded_char_list = pad_sequences(words.values, maxlen=900,
                                     padding='post',
                                     truncating='post')
    return padded_char_list


def get_train_val():
    print("pressing training data")
    train = pd.read_csv('../input/train_set.csv')
    train_char_arr = char_col_to_id_list(train)
    train_word_arr = word_col_to_id_list(train)
    train_y = to_categorical(train['class'].values - 1, num_classes=19)
    char_tra, char_val, word_tra, word_val, y_tra, y_val = train_test_split(
        train_char_arr,
        train_word_arr,
        train_y,
        train_size=0.92,
        random_state=233)
    del train, train_char_arr, train_word_arr, train_y
    print("pressinng testing data")
    test = pd.read_csv('../input/test_set.csv')
    char_test = char_col_to_id_list(test)
    word_test = word_col_to_id_list(test)
    del test
    np.savez_compressed("../input/train_validation_test_dataset.npz",
                        char_tra=char_tra,
                        char_val=char_val,
                        char_test=char_test,
                        word_tra=word_tra,
                        word_val=word_val,
                        word_test=word_test,
                        y_tra=y_tra,
                        y_val=y_val)
    print("file saved at input/train_validation_test_dataset.npz")
if __name__ == '__main__':
    get_train_val()