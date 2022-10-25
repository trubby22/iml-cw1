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
        if np.all(train_y, train_y[0]):
            return LeafNode(train_y[0]), depth
        else:
            label, attr, ldata_x, ldata_y, rdata_x, rdata_y = self.find_split(train_x, train_y)
            node = TreeNode(attr, label)
            node.left, ldepth = self.decision_tree_learning(ldata_x, ldata_y, depth + 1)
            node.right, rdepth = self.decision_tree_learning(rdata_x, rdata_y, depth + 1)
            return node, max(ldepth, rdepth)


    def find_split(self, train_x, train_y):
        max_info_gain = 0
        best_attr = None
        best_label = None
        best_lx = None
        best_ly = None
        best_rx = None
        best_ry = None
        for attr in range(train_x.shape[1]):
            info_gain, label, lx, ly, rx, ry = self.find_split_attr(train_x, train_y, attr)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_attr = attr
                best_label = label
                best_lx = lx
                best_ly = ly
                best_rx = rx
                best_ry = ry
        return best_label, best_attr, best_lx, best_ly, best_rx, best_ry

    def find_split_attr(self, train_x, train_y, attr):
        max_info_gain = 0
        best_label = None
        best_lx = None
        best_ly = None
        best_rx = None
        best_ry = None
        for label in range(2):
            lx, ly, rx, ry = self.split_data(train_x, train_y, attr, label)
            info_gain = self.information_gain(train_y, ly, ry)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_label = label
                best_lx = lx
                best_ly = ly
                best_rx = rx
                best_ry = ry
        return max_info_gain, best_label, best_lx, best_ly, best_rx, best_ry
    
    def split_data(self, train_x, train_y, attr, label):
        left_x = train_x[train_x[:, attr] == label]
        left_y = train_y[train_x[:, attr] == label]
        right_x = train_x[train_x[:, attr] != label]
        right_y = train_y[train_x[:, attr] != label]
        return left_x, left_y, right_x, right_y

    def information_gain(self, parent, left, right):
        p = len(left) / len(parent)
        return self.entropy(parent) - p * self.entropy(left) - (1 - p) * self.entropy(right)

    def entropy(self, data):
        if len(data) == 0:
            return 0
        p = np.sum(data) / len(data)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

if __name__ == '__main__':
    train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_y = np.array([0, 1, 1, 0])
    dtc = DecisionTreeCreator()
    tree, depth = dtc.decision_tree_learning(train_x, train_y, 0)
    print(depth)
