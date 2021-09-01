from sklearn import tree
from sklearn import metrics

class Arvore:
    def __init__(self, result = None, show = None):
        self._result = result
        self._show = show

    def treinamento_resultado(self, base, criterio):
        # Treinamendo da Árvore de Decisão
        model = tree.DecisionTreeClassifier(criterion=f"{self._criterio}")
        model = model.fit(base.get_x_train(), base.get_y_train())

        # Predição e Resultados

        result = model.predict(base.get_x_test())
        acc = metrics.accuracy_score(result, base.get_y_test())
        self._show = round(acc * 100)

    def __str__(self):
        return f"{self._show}%"

        print(list(result))
        print(list(y_test))