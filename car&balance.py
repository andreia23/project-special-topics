from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

url = "https://raw.githubusercontent.com/tmoura/machinelearning/master/datasets/iris.data"

# Carregar base de dados
dataset = pd.read_csv(url, header=None)

columns = len(dataset.columns)

y = dataset[0] # extrai a primeira coluna, que é o label
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
