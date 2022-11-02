from data_loader import *
from constants import *
from decision_tree_creator import *
from evaluator import *
from pruner import *


class Main:
    def __init__(self, path):
        self.path = path

    def run(self):
        dl = DataLoader(self.path)
        cross_validation = dl.generate_cross_validation_arr()
        trees = []
        eval_res = []
        pruned_trees = []
        pruned_eval_res = []
        p = Pruner()
        for training_set, validation_set, test_set in cross_validation:
            t = DecisionTreeCreator().learn(training_set)
            print('Unpruned tree')
            print(t)
            print()
            trees.append(t)
            e = Evaluator(test_data=test_set, tree=t)
            e.evaluate()
            print('Unpruned tree evaluation results')
            print(e)
            print()
            eval_res.append(e)
            pruned_t = p.prune(t, validation_set)
            print('Pruned tree')
            print(pruned_t)
            print()
            pruned_trees.append(pruned_t)
            e_pruned = Evaluator(test_data=test_set, tree=pruned_t)
            e_pruned.evaluate()
            print('Pruned tree evaluation results')
            print(e_pruned)
            print('', '-' * 20, '', sep='\n')
            pruned_eval_res.append(e_pruned)


if __name__ == '__main__':
    print('Clean data analysis')
    print()
    Main(clean_data_path).run()
    print('Noisy data analysis')
    print()
    Main(noisy_data_path).run()
