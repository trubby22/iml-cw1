from __future__ import annotations
from dataclasses import *


class Tree:
    def __init__(self, root: Node):
        self.root = root

    def visualise(self):
        pass

    def to_dict(self):
        return {}


class Node:
    def __init__(self, attribute, value, left, right):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right


if __name__ == '__main__':
    pass
