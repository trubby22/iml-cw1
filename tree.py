from __future__ import annotations
from dataclasses import *


class Tree:
    def __init__(self, root: Node, depth: int):
        self.root = root
        self.depth = depth

    def visualise(self):
        pass

    def to_dict(self):
        return {}


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


if __name__ == '__main__':
    pass
