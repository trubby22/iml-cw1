from tree import Tree
import numpy as np
from data_loader import DataLoader


class Evaluator:
    def __init__(self, test_data: np.ndarray, tree: Tree):
        self.test_data = test_data
        self.tree = tree
        self.c_matrix = None
        self.acc = None
        self.prec = None
        self.rec = None
        self.f1 = None

    def __repr__(self) -> str:
        return f'''
confusion matrix: 
{self.c_matrix}
accuracy: {self.acc:.0f}
precision: {self.prec}
recall: {self.rec}
f1_measure: {self.f1}
        '''.strip()

    def evaluate(self):
        self.confusion_matrix()
        self.accuracy()
        self.precision()
        self.recall()
        self.f1_measure()
        print(self)

    def confusion_matrix(self) -> np.ndarray:
        predicted_labels = self.tree.predict(self.test_data)
        actual_labels = np.array(self.test_data[:, -1])
        assert len(predicted_labels) == len(actual_labels)
        res = np.zeros(shape=(4, 4))
        for i in range(len(predicted_labels)):
            pl = int(predicted_labels[i])
            al = int(actual_labels[i])
            res[al - 1, pl - 1] += 1
        self.c_matrix = res
        return res

    def accuracy(self) -> float:
        correct = 0
        total = 0
        for i in range(4):
            for j in range(4):
                if i == j:
                    correct += self.c_matrix[i][j]
                total += self.c_matrix[i][j]
        res = correct / total
        self.acc = res
        return res

    def precision(self) -> np.ndarray:
        res = np.ndarray(shape=(4,))
        for i in range(4):
            correct = 0
            total = 0
            for j in range(4):
                if i == j:
                    correct = self.c_matrix[j][i]
                total += self.c_matrix[j][i]
            res[i] = correct / total
        self.prec = res
        return res

    def recall(self) -> np.ndarray:
        res = np.ndarray(shape=(4,))
        for i in range(4):
            correct = 0
            total = 0
            for j in range(4):
                if i == j:
                    correct = self.c_matrix[i][j]
                total += self.c_matrix[i][j]
            res[i] = correct / total
        self.rec = res
        return res

    def f1_measure(self):
        res = np.ndarray(shape=(4,))
        assert len(self.prec) == len(self.rec)
        for i in range(len(self.prec)):
            p = self.prec[i]
            r = self.rec[i]
            res[i] = 2 * p * r / (p + r)
        self.f1 = res
        return res


if __name__ == '__main__':
    dl = DataLoader()
    clean_data, noisy_data = dl.load_datasets()
    x: Tree = Tree.from_file()
    e = Evaluator(test_data=clean_data, tree=x)
    e.evaluate()
