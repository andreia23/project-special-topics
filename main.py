from treeD import Arvore
from k_nN import KnN
from trainTest import Train_test
import pandas as pd

def main():
    dataset = 'wine'
    randomBase = Train_test(dataset)
    y_position = 0
    x1_position = 1
    xn_position = 13

    randomBase.eighty_by_twenty(y_position, x1_position, xn_position)

    arvore_1 = Arvore(randomBase)
    arvore_1.treinamento_resultado('entropy')
    
    arvore_2 = Arvore(randomBase)
    arvore_2.treinamento_resultado('gini')

    knN_1 = KnN(randomBase)
    knN_1.treinamento_resultado('euclidean', 3)

    knN_2 = KnN(randomBase)
    knN_2.treinamento_resultado('euclidean', 3)
    
    knN_3 = KnN(randomBase)
    knN_3.treinamento_resultado('euclidean', 3)
    
    knN_4 = KnN(randomBase)
    knN_4.treinamento_resultado('manhattan', 3)

    knN_5 = KnN(randomBase)
    knN_5.treinamento_resultado('manhattan', 3)

    knN_6 = KnN(randomBase)
    knN_6.treinamento_resultado('manhattan', 3)

    print(arvore_1)
    print(arvore_2)

    print(knN_1)
    print(knN_2)
    print(knN_3)
    print(knN_4)
    print(knN_5)
    print(knN_6)

if __name__ == '__main__':
    main()