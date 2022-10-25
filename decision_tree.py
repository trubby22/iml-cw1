import numpy as np
from dataclasses import dataclass

@dataclass
class TreeNode:
    attr: int
    label: int
    left: 'TreeNode' | 'LeafNode'
    right: 'TreeNode' | 'LeafNode'

@dataclass
class LeafNode:
    def __init__(self, label):
        self.label = label

class DecisionTree:
    def __init__(self):
        self.root = None
        self.depth = 0

    def decision_tree_learning(self, train_x, train_y):
        self.root, self.depth = self.learn(train_x, train_y, 0)

    def learn(self, train_x, train_y, depth):
        if np.all(train_y == train_y[0]):
            return LeafNode(train_y[0]), depth
        else:
            label, attr, ldata_x, ldata_y, rdata_x, rdata_y = self.find_split(train_x, train_y)
            node = TreeNode(attr, label)
            node.left, ldepth = self.decision_tree_learning(ldata_x, ldata_y, depth + 1)
            node.right, rdepth = self.decision_tree_learning(rdata_x, rdata_y, depth + 1)
            return node, max(ldepth, rdepth)

    def find_split(self, train_x, train_y):
        pass

    def predict(self, test_x):
        curr = self.root

        while not isinstance(curr, LeafNode):
            if test_x[curr.label] < curr.attr:
                curr = curr.left
            else:
                curr = curr.right

        return curr.label
        


if __name__ == '__main__':
    pass
