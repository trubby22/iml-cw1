from __future__ import annotations
import numpy as np
from dataclasses import *
import pickle
import sys
from data_loader import *
from constants import *
from copy import deepcopy
from evaluator import *


class Tree:
    def __init__(self, root: Node, depth=None):
        self.root = root
        self.depth = depth

    def __str__(self):
        return str(self.root) 

    def to_file(self):
        sys.setrecursionlimit(1_000_000)
        with open('tree.pkl', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file() -> Tree:
        with open('tree.pkl', 'rb') as f:
            return pickle.load(f)

    def visualise(self):
        pass

    def to_dict(self) -> dict:
        return self.root.to_dict()

    def predict(self, text_x: np.ndarray) -> np.array:
        return np.array([self.root.predict(x) for x in text_x])

    def prune(self, test_x: np.ndarray) -> Tree:
        while True:
            old_root: Node
            new_root: Node
            old_root, new_root = self.root.prune()
            if not new_root:
                self.root = old_root
                return self
            old_tree = Tree(root=old_root, depth=self.depth)
            new_tree = Tree(root=new_root, depth=new_root.calc_depth())
            old_e = Evaluator(test_data=test_x, tree=old_tree)
            old_e.evaluate()
            old_accuracy = old_e.acc
            new_e = Evaluator(test_data=test_x, tree=new_tree)
            new_e.evaluate()
            new_accuracy = new_e.acc
            if new_accuracy >= old_accuracy:
                self.root = new_root
                self.depth = new_root.calc_depth()
            else:
                self.root = old_root


class Node:
    def __init__(
            self,
            split_value=None,
            attribute=None,
            left=None,
            right=None,
            label=None,
            cardinality=None,
            prune_tested=False
    ):
        """left < split_value <= right"""
        self.split_value = split_value
        self.attribute = attribute
        self.left: Node = left
        self.right: Node = right
        self.label = label
        self.cardinality = cardinality
        self.prune_tested = prune_tested

    def __str__(self):
        return self.to_str(0)

    def to_str(self, depth: int) -> str:
        if self.is_leaf_node:
            return f'{"  " * depth}label: {self.label}'

        return f'{"  " * depth}attribute: {self.attribute}, split_value: {self.split_value}\n' \
               f'{self.left.to_str(depth + 1)}\n' \
               f'{self.right.to_str(depth + 1)}'

    def __str__(self):
        return self.to_str(0)

    def to_str(self, depth: int) -> str:
        if self.is_leaf_node:
            return f'{"  " * depth}label: {self.label}'

        return f'{"  " * depth}attribute: {self.attribute}, split_value: {self.split_value}\n' \
               f'{self.left.to_str(depth + 1)}\n' \
               f'{self.right.to_str(depth + 1)}'

    @property
    def is_leaf_node(self):
        return not self.left and not self.right

    @property
    def is_leaf_parent(self):
        return (
                self.left and
                self.left.is_leaf_node and
                self.right and
                self.right.is_leaf_node
        )

    def to_dict(self) -> dict:
        if self.is_leaf_node:
            return {
                'label': self.label,
                'cardinality': self.cardinality,
            }
        return {
            'split_value': self.split_value,
            'attribute': self.attribute,
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    def predict(self, x: np.ndarray) -> int:
        if self.is_leaf_node:
            return self.label
        if x[self.attribute] < self.split_value:
            return self.left.predict(x)
        return self.right.predict(x)

    def prune(self):
        if self.is_leaf_node:
            return self, None
        if self.is_leaf_parent:
            old, new = self.prune_leaf_parent()
            return old, new
        l_old, l_new = self.left.prune()
        if l_new:
            old_node = deepcopy(self)
            old_node.left = l_old
            self.left = l_new
            return old_node, self
        r_old, r_new = self.right.prune()
        if r_new:
            old_node = deepcopy(self)
            old_node.right = r_old
            self.right = r_new
            return old_node, self
        return self, None

    def prune_leaf_parent(self):
        assert self.is_leaf_parent
        l_node: Node = self.left
        r_node: Node = self.right
        majority_node = l_node if l_node.cardinality >= r_node.cardinality else r_node
        majority_label = majority_node.label
        cardinality = l_node.cardinality + r_node.cardinality
        self.prune_tested = True
        return self, Node(label=majority_label, cardinality=cardinality)

    def calc_depth(self) -> int:
        if self.is_leaf_node:
            return 1
        l_depth = self.left.calc_depth()
        r_depth = self.right.calc_depth()
        return 1 + max(l_depth, r_depth)


if __name__ == '__main__':
    dl = DataLoader(clean_dataset)
    ds = dl.load_dataset()
    t = from_file('tree')
    res = t.predict(ds)
    print(res)

