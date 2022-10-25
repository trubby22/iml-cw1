import numpy as np

class TreeNode:
    def __init__(self, attr, label, left=None, right=None):
        self.attr = attr
        self.label = label
        self.left = left
        self.right = right

class LeafNode:
    def __init__(self, label):
        self.label = label

class DecisionTreeCreator:
    def __init__(self):
        pass

    def decision_tree_learning(self, train_x, train_y, depth):
        if np.all(train_y == train_y[0]):
            return LeafNode(train_y[0]), depth
        else:
            label, attr, ldata_x, ldata_y, rdata_x, rdata_y = self.find_split(train_x, train_y)
            node = TreeNode(attr, label)
            node.left, ldepth = self.decision_tree_learning(ldata_x, ldata_y, depth + 1)
            node.right, rdepth = self.decision_tree_learning(rdata_x, rdata_y, depth + 1)
            return node, max(ldepth, rdepth)


    def find_split(self):
        pass


if __name__ == '__main__':
    pass
