from datetime import datetime
import pickle
import sys


def timestamp():
    print(datetime.now().strftime('%H:%M:%S'))


def to_file(obj, filename):
    sys.setrecursionlimit(1_000_000)
    with open(f'{filename}.pkl', 'wb') as f:
        return pickle.dump(obj, f)


def from_file(filename):
    with open(f'{filename}.pkl', 'rb') as f:
        return pickle.load(f)
