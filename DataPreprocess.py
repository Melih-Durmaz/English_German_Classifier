import pandas as pd
import numpy as np



def processData():
    fp_german = open('train_german.txt', 'r')

    german_string = fp_german.read()

    german_string = german_string.replace('\n','')

    # Stringi 140'ar parçaya bölme.
    germanTrainingSet = np.array([german_string[i:i + 140] for i in range(0,len(german_string),140)])

    # germanTrainingSet = [german_string[i:i + 140] for i in range(0,len(german_string),140)]

    # Sondaki eleman 140'tan küçük olması ihitmaline karşı silinir
    np.delete(germanTrainingSet, len(germanTrainingSet) - 1, axis=0)

    # Test ve training setleri ayrılır
    germanTestSet = germanTrainingSet[2000:2200]
    germanTrainingSet = germanTrainingSet[0:2000]

    # 0'lardan oluşan bi dizi oluşturulur.
    german_labels = np.zeros((germanTrainingSet.__len__(),1))


    # Training ve test set'tekiarakterler ASCII karşılıklarına çevrilir.
    germanTrainingSet = [[ord(c[i]) for i in range(c.__len__())] for c in germanTrainingSet]
    germanTestSet = [[ord(c[i]) for i in range(c.__len__())] for c in germanTestSet]


    #print(germanTrainingSet[3])

    fp_german.close()

    # -----------
    '''
     İngilizcede de aynı işlemler yapılır
    '''
    fp_english = open('train_english.txt', 'r')

    english_string = fp_english.read()

    english_string = english_string.replace('\n','')

    englishTrainingSet = np.array([english_string[i:i + 140] for i in range(0,len(english_string),140)])


    # englishTrainingSet = [english_string[i:i + 140] for i in range(0,len(english_string),140)]

    np.delete(englishTrainingSet,len(englishTrainingSet) - 1, axis=0)

    englishTestSet = englishTrainingSet[2000:2200]


    englishTrainingSet = englishTrainingSet[0:2000]

    englishTrainingSet = [[ord(c[i]) for i in range(c.__len__())] for c in englishTrainingSet]

    englishTestSet = [[ord(c[i]) for i in range(c.__len__())] for c in englishTestSet]


    #print(englishTrainingSet.__len__())

    fp_english.close()

    return germanTrainingSet,englishTrainingSet,germanTestSet,englishTestSet


processData()