import numpy as np
from constants import *


_num_of_splits = 10
_training_split = 0.8
_validation_split = 0.1
_test_split = 0.1

assert _training_split + _validation_split + _test_split == 1


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.dataset = None
        self.cross_validation_arr = None

    def generate_cross_validation_arr(self) -> list[tuple]:
        self.load_dataset()
        self.shuffle()
        self.cross_validate()
        return self.cross_validation_arr

    def load_dataset(self):
        self.dataset = np.loadtxt(self.path)
        return self.dataset

    def shuffle(self):
        np.random.shuffle(self.dataset)

    def cross_validate(self):
        res = []
        self.dataset: np.ndarray
        sections = np.split(self.dataset, _num_of_splits)
        for i in range(_num_of_splits):
            training_ixs_raw = list(range(i, i + int(_training_split * _num_of_splits)))
            training_ixs = [j % _num_of_splits for j in training_ixs_raw]
            training_arr = [sections[j] for j in training_ixs]
            training_dataset = np.concatenate(training_arr)

            validation_ixs_raw = list(range(i, i + int(_validation_split * _num_of_splits)))
            validation_ixs = [j % _num_of_splits for j in validation_ixs_raw]
            validation_arr = [sections[j] for j in validation_ixs]
            validation_dataset = np.concatenate(validation_arr)

            test_ixs_raw = list(range(i, i + int(_test_split * _num_of_splits)))
            test_ixs = [j % _num_of_splits for j in test_ixs_raw]
            test_arr = [sections[j] for j in test_ixs]
            test_dataset = np.concatenate(test_arr)

            res.append((training_dataset, validation_dataset, test_dataset))
        self.cross_validation_arr = res


if __name__ == '__main__':
    dl = DataLoader(clean_data_path)
    clean_data = dl.load_dataset()
    print(clean_data)
    print(clean_data == 1)
