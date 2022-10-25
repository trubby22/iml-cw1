import numpy as np

class DataLoader:
    def __init__(self, file_name):
        data = np.loadtxt(file_name, dtype=np.float32)
        self.train, self.test = np.split(data, [int(0.67 * len(data))])
        self.train_x, self.train_y = np.split(self.train, [len(self.train[0]) - 1], axis=1)
        self.test_x, self.test_y = np.split(self.test, [len(self.test[0]) - 1], axis=1)

    def get_train_data(self):
        return self.train_x, self.train_y.astype(int)

    def get_test_data(self):
        return self.test_x, self.test_y.astype(int)


if __name__ == '__main__':
    loader = DataLoader("./wifi_db/clean_dataset.txt")
    train_x, train_y = loader.get_train_data()
    test_x, test_y = loader.get_test_data()
    print(train_x)
    print(train_y)
