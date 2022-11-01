from __future__ import annotations
from dataclasses import *


class Tree:
    def __init__(self, root: Node, depth: int):
        self.root = root
        self.depth = depth

    def visualise(self):
        pass

    def to_dict(self) -> dict:
        return self.root.to_dict()
        


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


if __name__ == '__main__':
    pass
