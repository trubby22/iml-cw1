from tree import Tree
import numpy as np


class Evaluator:
    def __init__(self):
        pass

    def evaluate(self):
        pass

    def confusion_matrix(self, test_data: np.ndarray, tree: Tree) -> np.ndarray:
        predicted_labels = tree.predict(test_data)
        actual_labels = np.array(test_data[:, -1])
        assert len(predicted_labels) == len(actual_labels)
        res = np.zeros(shape=(4, 4))
        for pl, al in zip(predicted_labels, actual_labels):
            res[al - 1, pl - 1] += 1
        return res

    def accuracy(self):
        pass

    def recall_and_precision_rates(self):
        pass

    def f1_measures(self):
        pass


if __name__ == '__main__':
    pass
