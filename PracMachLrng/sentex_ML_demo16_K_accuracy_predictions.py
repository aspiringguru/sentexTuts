'''
Final thoughts on K Nearest Neighbors - Practical Machine Learning Tutorial with Python p.19
https://youtu.be/r_D5TTV9-2c?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
'''
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups. IDIOT!!")
    distances = []
    for group in data:
        for features in data[group]:
            euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidian_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    #confidence = number of votes for the classification / k
    #print ("vote_result=", vote_result, "confidence=", confidence)
    return vote_result, confidence

accuracies = []
loopSize = 25

for i in range(loopSize):
    df = pd.read_csv('breast-cancer-wisconsin.data')
    #NB: pandas.read_csv interprets the frist row in file as column names.
    #print (df.head(3))
    #print (20*"#")
    df.replace('?', -99999, inplace = True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    #cast values to float since read_csv does not always produce desired result.
    #values as read in from csv should be int or float. convert to float to ensure known type.
    #print ("type(full_data)=", type(full_data))
    #print (full_data[:5])
    #print ("^before shuffle. ___________")
    random.shuffle(full_data)
    #print (full_data[:5])
    #print ("^after shuffle. ___________")
    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    #print ("len(full_data)=", len(full_data))
    train_data = full_data[:-int(test_size*len(full_data))]
    #print ("len(train_data)=", len(train_data))
    #train_data = row o to row (1-test_size) x num rows in full_data.
    test_data = full_data[-int(test_size*len(full_data)):]
    #print ("len(test_data)=", len(test_data))
    #560+139=699

    temp = 0
    for i in train_data:
        #print ("type(i)=", type(i), "i=", i)
        #each i is row of data as read in from file.
        #print ("i[-1]=", i[-1])
        #i[-1] = last column of the row of data is column 'class'
        train_set[i[-1]].append(i[:-1])
        #temp += 1
        #if temp>10: break
    #print ("train_set=", train_set)
    #print ("len(train_set)=", len(train_set))
    #print (type(train_set))#dict
    #print ("train.keys = ", list(train_set.keys()))
    #creates dictionary, keys = 2 for benign, 4 for malignant.
    #dictionary elements contain a list of lists made up of the data rows corresponding to the key (minus the key column).
    #i[-1] is the last column in the data - the 'class' column.
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0
    incorrect = 0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            #using k=5 since this is the default used in xxxx
            if group == vote:
                correct += 1
            else:
                #print (confidence)
                incorrect += 1
            total += 1
    #print ("No of incorrect votes = ", incorrect, " of total ", total, " votes.")
    #print ('Accuracy=', correct/total)
    accuracies.append(correct/total)

print ("loopSize=", loopSize, ", average accuracy = ", sum(accuracies)/len(accuracies))


#results for pre-loop runs.
#@k=25, Accuracy=0.9568, 0.9640, 0.94964, 0.9640
#@k=75, Accuracy=0.97122, 0.97122, 0.97122, 0.92805, 0.964028
#@k=75, Accuracy=0.94244, 0.92086, 0.92086, 0.97122, 0.94964
#recall num rows of data ~600, making k large % makes method less accurate.


'''
http://matplotlib.org/api/markers_api.html
https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)[source]
Matrix or vector norm.
This function is able to return one of eight different matrix norms, or one of an infinite number of vector norms
(described below), depending on the value of the ord parameter.
https://docs.python.org/3/howto/sorting.html
https://docs.python.org/3/library/functions.html#sorted

https://docs.python.org/2/library/collections.html#collections.Counter
most_common([n])
Return a list of the n most common elements and their counts from the most common to the least.

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
result : DataFrame or TextParser
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.astype.html
DataFrame.astype(dtype, copy=True, raise_on_error=True, **kwargs)
Cast object to input numpy.dtype Return a copy when copy = True (be really careful with this!)



'''