from data_loader import *
from constants import *
from decision_tree_creator import *
from evaluator import *
from pruner import *


if __name__ == '__main__':
    clean_dl = DataLoader(clean_data_path)
    clean_cross_validation = clean_dl.generate_cross_validation_arr()
    clean_trees = []
    clean_eval_res = []
    clean_pruned_trees = []
    clean_pruned_eval_res = []
    p = Pruner()
    for training_set, validation_set, test_set in clean_cross_validation:
        t = DecisionTreeCreator().learn(training_set)
        print('Unpruned tree')
        print(t)
        print()
        clean_trees.append(t)
        e = Evaluator(test_data=test_set, tree=t)
        e.evaluate()
        print('Unpruned tree evaluation results')
        print(e)
        print()
        clean_eval_res.append(e)
        pruned_t = p.prune(t, validation_set)
        print('Pruned tree')
        print(pruned_t)
        print()
        clean_pruned_trees.append(pruned_t)
        e_pruned = Evaluator(test_data=test_set, tree=pruned_t)
        e_pruned.evaluate()
        print('Pruned tree evaluation results')
        print(e_pruned)
        print('', '-' * 20, '', sep='\n')
        clean_pruned_eval_res.append(e_pruned)
