import numpy as np
from constants import *


_training_parts = 8
_validation_parts = 1
_test_parts = 1
_num_of_parts = _training_parts + _validation_parts + _test_parts


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
        sections = np.split(self.dataset, _num_of_parts)
        for i in range(_num_of_parts):
            start = i
            training_ixs_raw = list(range(
                start,
                start + _training_parts
            ))
            training_ixs = [j % _num_of_parts for j in training_ixs_raw]
            training_arr = [sections[j] for j in training_ixs]
            training_dataset = np.concatenate(training_arr)

            start += _training_parts
            validation_ixs_raw = list(range(
                start,
                start + _validation_parts
            ))
            validation_ixs = [j % _num_of_parts for j in validation_ixs_raw]
            validation_arr = [sections[j] for j in validation_ixs]
            validation_dataset = np.concatenate(validation_arr)

            start += _validation_parts
            test_ixs_raw = list(range(
                start,
                start + _test_parts
            ))
            test_ixs = [j % _num_of_parts for j in test_ixs_raw]
            test_arr = [sections[j] for j in test_ixs]
            test_dataset = np.concatenate(test_arr)

            combined = training_ixs + validation_ixs + test_ixs
            assert len(combined) == len(set(combined)), (training_ixs, validation_ixs, test_ixs)
            res.append((training_dataset, validation_dataset, test_dataset))
        self.cross_validation_arr = res


if __name__ == '__main__':
    dl = DataLoader(clean_data_path)
    clean_data = dl.load_dataset()
    print(clean_data)
    print(clean_data == 1)
