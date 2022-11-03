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
        avg_c_matrix = np.zeros(shape=(4,4))
        avg_acc = 0.0
        avg_prec = 0.0
        avg_rec = 0.0
        avg_f1 = 0.0
        avg_c_matrix_p = np.zeros(shape=(4,4))
        avg_acc_p = 0.0
        avg_prec_p = 0.0
        avg_rec_p = 0.0
        avg_f1_p = 0.0
        p = Pruner()
        size = 0
        for training_set, validation_set, test_set in cross_validation:
            size += 1
            t = DecisionTreeCreator().learn(training_set)
            print('Unpruned tree')
            print(t)
            print()
            trees.append(t)
            e = Evaluator(test_data=test_set, tree=t)
            c_matrix, acc, prec, rec, f1 = e.evaluate()
            avg_c_matrix += c_matrix
            avg_acc += acc
            avg_prec += prec
            avg_rec += rec
            avg_f1 += f1
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
            c_matrix, acc, prec, rec, f1 = e_pruned.evaluate()
            avg_c_matrix_p += c_matrix
            avg_acc_p += acc
            avg_prec_p += prec
            avg_rec_p += rec
            avg_f1_p += f1
            print('Pruned tree evaluation results')
            print(e_pruned)
            print('', '-' * 20, '', sep='\n')
            pruned_eval_res.append(e_pruned)
        avg_c_matrix /= size
        avg_acc /= size
        avg_prec /= size
        avg_rec /= size
        avg_f1 /= size
        avg_c_matrix_p /= size
        avg_acc_p /= size
        avg_prec_p /= size
        avg_rec_p /= size
        avg_f1_p /= size
        print('Unpruned averages:')
        print()
        print(f'''
confusion matrix: 
{avg_c_matrix}
accuracy: {avg_acc}
precision: {[round(x, 2) for x in avg_prec]}
recall: {[round(x, 2) for x in avg_rec]}
f1 measure: {[round(x, 2) for x in avg_f1]}
                ''')
        print()
        print('Pruned averages:')
        print()
        print(f'''
confusion matrix: 
{avg_c_matrix_p}
accuracy: {avg_acc_p}
precision: {[round(x, 2) for x in avg_prec_p]}
recall: {[round(x, 2) for x in avg_rec_p]}
f1 measure: {[round(x, 2) for x in avg_f1_p]}
                ''')

if __name__ == '__main__':
    print('Clean data analysis')
    print()
    Main(clean_data_path).run()
    print('Noisy data analysis')
    print()
    Main(noisy_data_path).run()
