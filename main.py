from data_loader import DataLoader
from decision_tree import DecisionTree
from evaluator import Evaluator
from pruner import Pruner


if __name__ == '__main__':
    loader = DataLoader("./wifi_db/clean_dataset.txt")
    train_x, train_y, test_x, test_y = loader.load_data()

    tree = DecisionTree()
    tree.decision_tree_learning(train_x, train_y)

    evaluator = Evaluator()
    evaluator.evaluate(test_x, test_y, tree)

    pruner = Pruner()
    pruned_tree = pruner.prune(tree)
