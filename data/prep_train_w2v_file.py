# utf-8
import pandas as pd
import numpy as np

train = pd.read_csv('../input/train_set.csv', usecols=['article', 'word_seg'])
test = pd.read_csv('../input/test_set.csv', usecols=['article', 'word_seg'])

df: pd.DataFrame = pd.concat([train, test])
del train, test
df.to_csv('../input/Char.txt', columns=['article'], header=False, index=False)
df.to_csv('../input/word.txt', columns=['word_seg'], header=False, index=False)
