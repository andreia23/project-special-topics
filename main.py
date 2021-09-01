from treeD import Arvore
from trainTest import Train_test
import pandas as pd

def main():
    randomBase_1 = Train_test('iris')
    randomBase_1.eighty_by_twenty()

    arvore_1 = Arvore().treinamento_resultado(randomBase, 'entropy')
    #arvore_2 = Arvore(randomBase, 'gini')

    #knN_1 = KnN(randomBase, 'euclidian', 3)
    #knN_2 = KnN(randomBase, 'euclidian', 3)
    #knN_3 = KnN(randomBase, 'euclidian', 3)
    #knN_4 = KnN(randomBase, 'vdm', 3)
    #knN_5 = KnN(randomBase, 'vdm', 3)
    #knN_6 = KnN(randomBase, 'vdm', 3)

    #print(arvore_1)
    #print(arvore_2)

    #print(knN_1)
    #print(knN_2)
    #print(knN_3)
    #print(knN_4)
    #print(knN_5)
    #print(knN_6)

if __name__ == '__main__':
    main()