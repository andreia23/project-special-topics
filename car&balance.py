from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

url_1 = open("datasets/balance-scale.data","r+")
url_2 = open("datasets/balance-scale.data","r+")

# Carregar base de dados
dataset_1 = pd.read_csv(url_1, header=None)
dataset_2 = pd.read_csv(url_2, header=None)

columns = len(dataset.columns)

y = dataset[0]
X = dataset.loc[:,1:columns-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

# Treinamendo da Árvore de Decisão
model = tree.DecisionTreeClassifier(criterion="entropy")
model = model.fit(X_train, y_train)

# Predição e Resultados

result = model.predict(X_test)
acc = metrics.accuracy_score(result, y_test)
show = round(acc * 100)
print("{}%".format(show))

print(list(result))
print(list(y_test))
