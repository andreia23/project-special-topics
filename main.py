from treeD import Arvore
from k_nN import KnN
from datasets_ import Dataset
from trainTest import Train_test
import pandas as pd

#Parameters: Name, Y_position, Number of Attributes
#Dataset("wine", "inicial", 13) 
#Dataset("iris", "final", 4) 
#Dataset("balance-scale", "inicial", 4) 

def main():
    dataset = Dataset("iris", "final", 4)
    randomBase = Train_test(dataset)
    randomBase.eighty_by_twenty()

    arvore_1 = Arvore(randomBase)
    arvore_1.treinamento_resultado('entropy')
    
    #arvore_2 = Arvore(randomBase)
    #arvore_2.treinamento_resultado('gini')

    #knN_1 = KnN(randomBase)
    #knN_1.treinamento_resultado('euclidean', 3)

    #knN_2 = KnN(randomBase)
    #knN_2.treinamento_resultado('euclidean', 3)
    
    #knN_3 = KnN(randomBase)
    #knN_3.treinamento_resultado('euclidean', 3)
    
    #knN_4 = KnN(randomBase)
    #knN_4.treinamento_resultado('manhattan', 3)

    #knN_5 = KnN(randomBase)
    #knN_5.treinamento_resultado('manhattan', 3)

    #knN_6 = KnN(randomBase)
    #knN_6.treinamento_resultado('manhattan', 3)

    print(arvore_1)
    #print(arvore_2)

    #print(knN_1)
    #print(knN_2)
    #print(knN_3)
    #print(knN_4)
    #print(knN_5)
    #print(knN_6)

if __name__ == '__main__':
    main()