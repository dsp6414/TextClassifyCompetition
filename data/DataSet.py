import numpy as np
import os
import random
from torch.utils import data


class TrainData(data.Dataset):
    """
    导入训练验证集
    """

    def __init__(self, data_arr, label_arr, training=True, augment=True):
        assert data_arr.shape[0] == label_arr.shape[0]
        self.data = data_arr
        self.label = label_arr
        self.augment = augment
        self.training = training

    def shuffle(self, d):
        return np.random.permutation(d.tolist())

    def dropout(self, d, p=0.3):
        len_ = len(d)
        index = np.random.choice(len_, int(len_ * p))
        d[index] = 0
        return d

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        s, t = self.data[item], self.label[item]
        if self.augment and self.training:
            a = random.random()
            if a > 0.7:
                s = self.dropout(s, 0.3)
            elif a > 0.4:
                s = self.shuffle(s)

        return s, t


class TestData(data.Dataset):
    """
    导入训练验证集
    """

    def __init__(self, data_arr):
        self.data = data_arr

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        s = self.data[item]
        return s
