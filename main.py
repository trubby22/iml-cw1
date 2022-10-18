from data_loader import DataLoader
from decision_tree_creator import DecisionTreeCreator
from evaluator import Evaluator
from pruner import Pruner


if __name__ == '__main__':
    loader = DataLoader()
    data = loader.load_data()
    creator = DecisionTreeCreator()
    depth = None
    tree = creator.decision_tree_learning(data, depth)
    evaluator = Evaluator()
    evaluator.evaluate(tree)
    pruner = Pruner()
    pruned_tree = pruner.prune(tree)
