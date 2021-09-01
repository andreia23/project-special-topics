from sklearn.model_selection import train_test_split
import pandas as pd

class Train_test:
    def __init__(self, filename, x_train = None, x_test = None, y_train = None, y_test = None):
        self._X_train = x_train
        self._X_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._filename = filename

    def get_x_train(self):
        return self._X_train

    def get_x_test(self):
        return self._X_test

    def get_y_train(self):
        return self._y_train

    def get_y_test(self):
        return self._y_test

    def eighty_by_twenty(self, index_Y, index_inicial, index_final):
        arquivo = open(f"datasets/{self._filename}.data","r+")
    
        dataset = pd.read_csv(arquivo, header=None)

        columns = len(dataset.columns)

        y = dataset[index_Y]
        X = dataset.loc[:,index_inicial:index_final]

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

    def __str__(self):
        return "Train and Test"