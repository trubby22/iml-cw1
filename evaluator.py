import numpy as np
from data_loader import DataLoader
from constants import *
from utils import *
import math


class Evaluator:
    def __init__(self, test_data: np.ndarray, tree):
        self.test_data = test_data
        self.tree = tree
        self.c_matrix = None
        self.acc = None
        self.prec = None
        self.rec = None
        self.f1 = None

    def __repr__(self):
        return f'''
confusion matrix: 
{self.c_matrix}
accuracy: {self.acc:.2f}
precision: {[float(f'{x:.2f}') for x in self.prec]}
recall: {[float(f'{x:.2f}') for x in self.rec]}
f1_measure: {[float(f'{x:.2f}') for x in self.f1]}
        '''.strip()

    def evaluate(self):
        self.confusion_matrix()
        self.accuracy()
        self.precision()
        self.recall()
        self.f1_measure()
        return self.c_matrix, self.acc, self.prec, self.rec, self.f1 

    def confusion_matrix(self):
        predicted_labels = self.tree.predict(self.test_data)
        actual_labels = np.array(self.test_data[:, -1])
        assert len(predicted_labels) == len(actual_labels)
        res = np.zeros(shape=(4, 4))
        for i in range(len(predicted_labels)):
            pl = int(predicted_labels[i])
            al = int(actual_labels[i])
            res[al - 1, pl - 1] += 1
        self.c_matrix = res

    def accuracy(self):
        correct = 0
        total = 0
        for i in range(4):
            for j in range(4):
                if i == j:
                    correct += self.c_matrix[i][j]
                total += self.c_matrix[i][j]
        res = correct / total if total != 0 else math.nan
        self.acc = res

    def precision(self):
        res = np.ndarray(shape=(4,))
        for i in range(4):
            correct = 0
            total = 0
            for j in range(4):
                if i == j:
                    correct = self.c_matrix[j][i]
                total += self.c_matrix[j][i]
            res[i] = correct / total if total != 0 else math.nan
        self.prec = res

    def recall(self):
        res = np.ndarray(shape=(4,))
        for i in range(4):
            correct = 0
            total = 0
            for j in range(4):
                if i == j:
                    correct = self.c_matrix[i][j]
                total += self.c_matrix[i][j]
            res[i] = correct / total if total != 0 else math.nan
        self.rec = res

    def f1_measure(self):
        res = np.ndarray(shape=(4,))
        assert len(self.prec) == len(self.rec)
        for i in range(len(self.prec)):
            p = self.prec[i]
            r = self.rec[i]
            res[i] = 2 * p * r / (p + r)
        self.f1 = res


if __name__ == '__main__':
    from decision_tree_creator import DecisionTreeCreator
    timestamp()
    dl = DataLoader(noisy_data_path)
    arr = dl.generate_cross_validation_arr()
    timestamp()
    trained_trees = []
    for training_set, validation_set, test_set in arr:
        t = DecisionTreeCreator().learn(training_set)
        trained_trees.append(t)
        e = Evaluator(test_data=test_set, tree=t)
        e.evaluate()
        print(e)
        timestamp()
    to_file(trained_trees, 'trees')
