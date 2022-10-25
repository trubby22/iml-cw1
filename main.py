from data_loader import DataLoader
from decision_tree_creator import DecisionTreeCreator
from evaluator import Evaluator
from pruner import Pruner


if __name__ == '__main__':
    loader = DataLoader("./wifi_db/clean_dataset.txt")
    train_x, train_y, test_x, test_y = loader.load_data()

    creator = DecisionTreeCreator()
    depth = None
    tree = creator.decision_tree_learning(train_x, train_y, depth)

    evaluator = Evaluator()
    evaluator.evaluate(tree)

    pruner = Pruner()
    pruned_tree = pruner.prune(tree)
