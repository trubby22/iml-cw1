from typing import Tuple, Any

import numpy as np
from tree import *
from data_loader import *
import math


class DecisionTreeCreator:
    def __init__(self):
        self.attribute_ixs = range(7)
        self.label_ix = 7
        self.labels = range(1, 5)

    def decision_tree_learning(self, training_dataset: np.ndarray, depth):
        if self.check_if_all_samples_have_same_label(training_dataset):
            label = training_dataset[:, self.label_ix][0]
            return Node(label=label), depth
        else:
            attribute_ix, split_value = self.find_split(training_dataset)
            node = Node(split_value=split_value, attribute=attribute_ix)
            l_dataset, r_dataset = np.split(training_dataset, split_value)
            l_branch, l_depth = self.decision_tree_learning(l_dataset, depth + 1)
            r_branch, r_depth = self.decision_tree_learning(r_dataset, depth + 1)
            return node, max(l_depth, r_depth)

    def check_if_all_samples_have_same_label(self, dataset: np.ndarray):
        labels = dataset[:, self.label_ix]
        return np.all(labels == labels[0])

    def find_split(self, training_dataset: np.ndarray) -> tuple[int, int]:
        res = []
        for ix in self.attribute_ixs:
            training_dataset.sort(axis=0, order=f'A{ix}')
            for row in training_dataset:
                split_value = row[ix]
                s_left, s_right = np.split(training_dataset, split_value)
                information_gain = self.calculate_information_gain(training_dataset, s_left, s_right)
                res.append((information_gain, ix, split_value))
        res.sort(key=lambda xyz: xyz[0], reverse=True)
        assert res
        _, attribute_ix, value = res[0]
        return attribute_ix, value

    def calculate_information_gain(self, s_all, s_left, s_right):
        return self.entropy(s_all) - self.remainder(s_left, s_right)

    def entropy(self, dataset: np.ndarray):
        labels = [0, 0, 0, 0]
        total = 0
        for row in dataset:
            label = row[self.label_ix]
            labels[label - 1] += 1
            total += 1
        ps = [x / total for x in labels]
        log_ps = [p * math.log2(p) for p in ps]
        return - sum(log_ps)

    def remainder(self, s_left: np.ndarray, s_right: np.ndarray):
        c_left = 0
        c_right = 0
        c_all = c_left + c_right
        return (c_left * self.entropy(s_left) + c_right * self.entropy(s_right)) / c_all


if __name__ == '__main__':
    dl = DataLoader()
    clean_data, noisy_data = dl.load_datasets()
    dtc = DecisionTreeCreator()
    node, depth = dtc.decision_tree_learning(clean_data, 0)
    print(node)
    print(depth)
