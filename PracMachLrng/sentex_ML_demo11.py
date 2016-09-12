'''
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

'''
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
#we know from breast-cancer-wisconsin.names that 'Missing attribute values: 16'
#need to replace '?' with a nominated value which will not crash and can be filtered out.
print (type(df))
df.replace("?", -99999, inplace=True)
#makes the missing data a hugh outlier
df.drop(['id'], 1, inplace=True)
#drop the id column as it does not aid prediction in any way.

X = np.array(df.drop(['class'], 1))
#drop the 'class' column from X as we will be predicting 'class'
y = np.array(df['class'])
#use 'class' column for y as this is the column we want to predict.

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
#split X into train/test data sets.

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print ("accuracy=", accuracy)

#example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
#make up some data to predict from when we use the classifier created.
print ("len(example_measures)=", len(example_measures))
example_measures = example_measures.reshape(len(example_measures), -1)
#use len(example_measures) to enable any size of input data to be used.
prediction = clf.predict(example_measures)
print ("prediction=", prediction, "type(prediction)=", type(prediction))
print (df.head())


'''
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

NB: dropping the 'id' column improves the accuracy. id values do not add value to prediction.
'''