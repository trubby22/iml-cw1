import numpy as np
from constants import *


class DataLoader:
    def __init__(self):
        pass

    def load_datasets(self):
        return np.loadtxt(clean_dataset), np.loadtxt(noisy_dataset)

    def load_datasets_old(self):
        return self.load_data(clean_dataset), self.load_data(noisy_dataset)

    @staticmethod
    def load_data(path):
        dtype = list(zip([f'A{x}' for x in range(7)], [np.float32] * 7))
        dtype.append(('L', np.int32))
        return np.loadtxt(path, dtype=dtype)

    def generate_training_and_test_datasets(self):
        pass


if __name__ == '__main__':
    dl = DataLoader()
    clean_data, noisy_data = dl.load_datasets()
    print(clean_data)
    print(noisy_data)
    print(clean_data == 1)
