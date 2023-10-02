import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        if(len(labels.shape) == 2):
            self.labels = labels.reshape(labels.shape[0], -1, labels.shape[1])
            # print(self.labels.shape)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.shape[0]

    @property
    def num_input_features(self):
        return self.features.shape[-1]

    @property
    def num_output_features(self):
        return self.labels.shape[-1]

if __name__ == '__main__':
    print("dataset_seq2seq")