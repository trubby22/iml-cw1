from __future__ import annotations
import numpy as np
from dataclasses import *
import numpy as np


class Tree:
    def __init__(self, root: Node, depth: int):
        self.root = root
        self.depth = depth

    def visualise(self):
        pass

    def to_dict(self) -> dict:
        return self.root.to_dict()

    def predict(self, text_x: np.ndarray) -> np.array:
        return np.array([self.root.predict(x) for x in text_x])


class Node:
    def __init__(self, split_value=None, attribute=None, left=None, right=None, label=None):
        """left < split_value <= right"""
        self.split_value = split_value
        self.attribute = attribute
        self.left = left
        self.right = right
        self.label = label

    @property
    def is_leaf_node(self):
        return not self.left and not self.right

    def to_dict(self) -> dict:
        if self.is_leaf_node:
            return {'label': self.label}
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

if __name__ == '__main__':
    pass
