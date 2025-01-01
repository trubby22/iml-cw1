from __future__ import annotations
import numpy as np
from dataclasses import *
import pickle
import sys
from data_loader import *
from constants import *
from evaluator import *

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class Tree:
    def __init__(self, root: Node, depth=None):
        self.root = root
        self.depth = depth

    # def __str__(self):
    #     return str(self.root) 

    def to_file(self):
        sys.setrecursionlimit(1_000_000)
        with open('tree.pkl', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file():
        with open('tree.pkl', 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        return f'''
Tree with depth: {self.depth}
Total number of nodes: {self.root.cardinality}
Number of leaf nodes: {self.root.calc_leaf_nodes()}
Number of non-leaf nodes: {self.root.cardinality - self.root.calc_leaf_nodes()}
        '''.strip()

    def to_dict(self):
        return self.root.to_dict()

    def predict(self, test_x: np.ndarray):
        return np.array([self.root.predict(x) for x in test_x])

    def prune(self, validation_set: np.ndarray):
        self.root, _ = self.root.prune(self.root, validation_set)
        pruned_depth = self.root.calc_depth()
        assert pruned_depth <= self.depth, (self.depth, pruned_depth)
        self.depth = pruned_depth
        return self

    def visualise(self):
        NODE_DIST = 50
        DEPTH_DIST = 3

        lines = []
        labels = []

        fig, ax = plt.subplots()

        def draw_node(node, x = 0, y = 0, width = NODE_DIST):
            xl = x + DEPTH_DIST
            yl = y - width / 0.3
            xr = x + DEPTH_DIST
            yr = y + width / 0.3

            if node.is_leaf_node:
                labels.append((str(node), (x, y), True))
            else:
                lines.append([[x, y], [xl, yl]])
                lines.append([[x, y], [xr, yr]])
                labels.append((str(node), (x, y), False))
                draw_node(node.left, xl, yl, width / 2)
                draw_node(node.right, xr, yr, width  / 2)

        draw_node(self.root)

        ax.set_xlim(-1, self.depth * DEPTH_DIST + 1)
        ax.set_ylim(-7 * NODE_DIST, 7 * NODE_DIST)

        line_collection = LineCollection(lines)
        ax.add_collection(line_collection)

        for label, (x, y), isLeaf in labels:
            ax.annotate(label, (x, y), ha='center', va='center', size=7, color="white",bbox=dict(boxstyle="round,pad=0.2", fc= "purple" if isLeaf else "blue", lw=0))

        plt.show()


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

    def to_str(self, depth: int):
        if self.is_leaf_node:
            return f'{"  " * depth}label: {self.label}'

        return f'{"  " * depth}attribute: {self.attribute}, split_value: {self.split_value}\n' \
               f'{self.left.to_str(depth + 1)}\n' \
               f'{self.right.to_str(depth + 1)}'

    def __str__(self):
        return self.to_str(0)

    def to_str(self, depth: int):
        if self.is_leaf_node:
            return f'{"  " * depth}label: {self.label}'

        return f'{"  " * depth}attribute: {self.attribute}, split_value: {self.split_value}\n' \
               f'{self.left.to_str(depth + 1)}\n' \
               f'{self.right.to_str(depth + 1)}'

    @property
    def is_leaf_node(self):
        return not self.left and not self.right

    @property
    def can_be_pruned(self):
        return (
            self.is_leaf_parent and
            not self.prune_tested
        )

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

    @staticmethod
    def predict_root(root, test_x: np.ndarray):
        return np.array([root.predict(x) for x in test_x])

    def predict(self, x: np.ndarray):
        if self.is_leaf_node:
            return self.label
        if x[self.attribute] < self.split_value:
            return self.left.predict(x)
        return self.right.predict(x)

    def prune(self, root: Node, validation_set: np.ndarray):
        if self.is_leaf_node:
            return self, None
        l_old, l_new = self.left.prune(root, validation_set)
        if l_new:
            self.left = l_old
            old_t = Tree(root)
            old_e = Evaluator(validation_set, old_t)
            old_e.evaluate()
            old_acc = old_e.acc

            self.left = l_new
            new_t = Tree(root)
            new_e = Evaluator(validation_set, new_t)
            new_e.evaluate()
            new_acc = new_e.acc

            if new_acc >= old_acc:
                self.left = l_new
            else:
                self.left = l_old
        r_old, r_new = self.right.prune(root, validation_set)
        if r_new:
            self.right = r_old
            old_t = Tree(root)
            old_e = Evaluator(validation_set, old_t)
            old_e.evaluate()
            old_acc = old_e.acc

            self.right = r_new
            new_t = Tree(root)
            new_e = Evaluator(validation_set, new_t)
            new_e.evaluate()
            new_acc = new_e.acc

            if new_acc >= old_acc:
                self.right = r_new
            else:
                self.right = r_old
        if self.can_be_pruned:
            return self.prune_leaf_parent()
        return self, None

    def prune_leaf_parent(self):
        assert self.can_be_pruned
        l_node: Node = self.left
        r_node: Node = self.right
        majority_node = l_node if l_node.cardinality >= r_node.cardinality else r_node
        majority_label = majority_node.label
        cardinality = l_node.cardinality + r_node.cardinality
        self.prune_tested = True
        return self, Node(label=majority_label, cardinality=cardinality)

    def calc_depth(self) -> int:
        if self.is_leaf_node:
            return 0
        l_depth = self.left.calc_depth()
        r_depth = self.right.calc_depth()
        return 1 + max(l_depth, r_depth)

    def calc_leaf_nodes(self):
        if self.is_leaf_node:
            return 1
        l_leaves = self.left.calc_leaf_nodes()
        r_leaves = self.right.calc_leaf_nodes()
        return l_leaves + r_leaves

    def __repr__(self):
        if self.is_leaf_node:
            return f'''{self.label}'''
        else:
            return f'''{self.attribute} > {self.split_value}'''


if __name__ == '__main__':
    dl = DataLoader(clean_data_path)
    ds = dl.load_dataset()
    t = from_file('tree')
    res = t.predict(ds)
    print(res)

    t.visualise()
