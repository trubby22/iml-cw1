import math

from data_loader import *
from tree import Tree, Node
from utils import *


class DecisionTreeCreator:
    def __init__(self):
        self.attribute_ixs = np.arange(7)
        self.label_ix = -1
        self.labels = np.arange(1, 5)

    def learn(self, training_dataset):
        root, depth = self.decision_tree_learning(training_dataset, 0)
        return Tree(root=root, depth=depth)

    def decision_tree_learning(self, training_dataset: np.ndarray, depth):
        labels = training_dataset[:, self.label_ix]
        if np.unique(labels).size == 1:
            return Node(label=labels[0], cardinality=labels.size), depth
        else:
            attribute_ix, split, split_value = self.find_split(training_dataset)
            training_dataset = training_dataset[training_dataset[:, attribute_ix].argsort()]
            l_dataset, r_dataset = np.split(training_dataset, [split])
            l_branch, l_depth = self.decision_tree_learning(l_dataset, depth + 1)
            r_branch, r_depth = self.decision_tree_learning(r_dataset, depth + 1)
            node = Node(
                split_value=split_value,
                attribute=attribute_ix,
                left=l_branch,
                right=r_branch,
                cardinality=l_branch.cardinality + r_branch.cardinality
            )
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

    def max_info_gain(self, dataset: np.ndarray, attribute_ix) -> tuple[float, int]:
        max_info_gain = -math.inf
        best_split = None
        for split in np.arange(1, dataset.shape[0]):
            if dataset[split - 1, attribute_ix] == dataset[split, attribute_ix]:
                continue
            info_gain = self.information_gain(dataset, split)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_split = split
        return max_info_gain, best_split

    def information_gain(self, dataset: np.ndarray, split: int) -> float:
        l_dataset, r_dataset = np.split(dataset, [split])
        return self.entropy(dataset) - self.entropy(l_dataset) - self.entropy(r_dataset)

    def entropy(self, dataset: np.ndarray) -> float:
        entropy = 0
        for label in np.unique(dataset[:, self.label_ix]):
            p = self.probability(dataset[:, self.label_ix], label)
            entropy += p * np.log2(p)
        return -entropy

    @staticmethod
    def probability(dataset: np.array, label: int) -> float:
        return dataset[dataset == label].shape[0] / dataset.shape[0]


if __name__ == '__main__':
    dl = DataLoader(clean_dataset)
    clean_data = dl.load_dataset()
    dtc = DecisionTreeCreator()
    timestamp()
    node, depth = dtc.decision_tree_learning(clean_data, 0)
    timestamp()
    tree = Tree(node, depth)
    print(depth)
    labels = clean_data[:, -1]
    prediction = tree.predict(clean_data[:, :-1])
    print(len(labels[labels == prediction]) / len(labels))
    to_file(tree, 'tree')
    x = from_file('tree')
    print(x)
