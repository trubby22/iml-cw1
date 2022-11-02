from tree import Tree
import numpy as np
from data_loader import *
from utils import *
import sys


class Pruner:
    def __init__(self):
        pass

    def prune(self, tree: Tree, validation_set: np.ndarray):
        sys.setrecursionlimit(1_000_000)
        return tree.prune(validation_set)


if __name__ == '__main__':
    p = Pruner()
    dl = DataLoader(clean_dataset)
    arr = dl.generate_cross_validation_arr()
    trees = from_file('trees')
    pruned_trees = []
    for tree, datasets in zip(trees, arr):
        train_set, validation_set, test_set = datasets
        pruned_tree = p.prune(tree, validation_set)
        pruned_trees.append(pruned_tree)
    to_file(pruned_trees, 'pruned_trees')
