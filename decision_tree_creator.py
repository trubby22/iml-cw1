from typing import Tuple, Any

import numpy as np
from tree import *
from data_loader import *
import math
from datetime import datetime
from utils import *


class DecisionTreeCreator:
    def __init__(self):
        self.attribute_ixs = range(7)
        self.label_ix = -1
        self.labels = range(1, 5)

    def learn(self, training_dataset):
        root, depth = self.decision_tree_learning(training_dataset, 0)
        return Tree(root=root, depth=depth)

    def decision_tree_learning(self, training_dataset: np.ndarray, depth):
        if len(set(training_dataset[:, self.label_ix])) == 1:
            label = training_dataset[:, self.label_ix][0]
            return Node(label=label), depth
        else:
            attribute_ix, split, split_value = self.find_split(training_dataset)
            training_dataset = training_dataset[training_dataset[:, attribute_ix].argsort()]
            l_dataset, r_dataset = np.split(training_dataset, [split])
            l_branch, l_depth = self.decision_tree_learning(l_dataset, depth + 1)
            r_branch, r_depth = self.decision_tree_learning(r_dataset, depth + 1)
            node = Node(split_value=split_value, attribute=attribute_ix, left=l_branch, right=r_branch)
            return node, max(l_depth, r_depth)

    def find_split(self, dataset: np.ndarray):
        max_info_gain = -math.inf
        split_value = None
        best_split = None
        attribute_ix = None
        for ix in self.attribute_ixs:
            dataset = dataset[dataset[:, ix].argsort()]
            info_gain, split = self.max_info_gain(dataset, ix)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_split = split
                split_value = dataset[split][ix]
                attribute_ix = ix
        return attribute_ix, best_split, split_value

    def max_info_gain(self, dataset: np.ndarray, attribute_ix: int) -> Tuple[float, int]:
        max_info_gain = -math.inf
        best_split = None
        for split in range(1, len(dataset)):
            info_gain = self.information_gain(dataset, attribute_ix, split)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_split = split
        return max_info_gain, best_split

    def information_gain(self, dataset: np.ndarray, attribute_ix: int, split: int) -> float:
        l_dataset, r_dataset = np.split(dataset, [split])
        return self.entropy(dataset) - self.entropy(l_dataset) - self.entropy(r_dataset)
    
    def entropy(self, dataset: np.ndarray) -> float:
        entropy = 0
        for label in set(dataset[:, self.label_ix]):
            p = self.probability(dataset[:, self.label_ix], label)
            entropy += p * np.log2(p)
        return -entropy

    def probability(self, dataset: np.array, label: int) -> float:
        return len(dataset[dataset == label]) / len(dataset)


if __name__ == '__main__':
    dl = DataLoader()
    clean_data, noisy_data = dl.load_dataset()
    dtc = DecisionTreeCreator()
    timestamp()
    node, depth = dtc.decision_tree_learning(clean_data, 0)
    timestamp()
    tree = Tree(node, depth)
    print(depth)
    labels = clean_data[:, -1]
    prediction = tree.predict(clean_data[:, :-1])
    print(len(labels[labels == prediction]) / len(labels))
    tree.to_file()
    x = Tree.from_file()
    print(x)
