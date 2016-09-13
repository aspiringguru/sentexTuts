'''
Support Vector Machine Intro and Application - Practical Machine Learning Tutorial with Python p.20
https://youtu.be/mA5nwGoRAOo?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

'''
import time
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm

import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

confidences = []
predictions = []
start_time = time.time()
for i in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    #clf = neighbors.KNeighborsClassifier()
    clf = svm.SVC()


    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(confidence)
    confidences.append(confidence)

    example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
    example_measures = example_measures.reshape(len(example_measures), -1)
    prediction = clf.predict(example_measures)
    print(prediction, type(float(list(prediction)[0])))
    predictions.append(float(list(prediction)[0]))

print ("confidences=", confidences)
print ("predictions=", predictions)
print ("average confidence=", sum(confidences)/len(confidences))
print("--- %s seconds ---" % (time.time() - start_time))
"""

"""