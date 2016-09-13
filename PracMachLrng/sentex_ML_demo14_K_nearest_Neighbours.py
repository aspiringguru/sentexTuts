'''
Creating Our K Nearest Neighbors Algorithm - Practical Machine Learning with Python p.16 & 17
https://youtu.be/n3RqsMz3-0A?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

K nearest neighbours comparison requires _ALL_ points compared.
big O notation assessment is huge. ie: slow and does not scale well.


https://en.wikipedia.org/wiki/Euclidean_distance
The distance (d) from p to q, or from q to p is given by the Pythagorean formula:
d(q,p) = d(p,q) = sqrt( (q1-p1)^2 + (q2-p2)^2 + .... +  (qn-pn)^2)
[recall hyptoneuse of 90 deg triangle formula h = sqrt(x^2 + y^2) where x & y are the square sides.]
euclidian distance = sqrt(Sum [i=1 to n]  (qi - pi)^2)
'''
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use("fivethirtyeight")


dataset = { 'k': [[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}
print ("type(dataset)=", type(dataset))
#added a second new_features to demonstrate the k_nearest_neighbors result.
new_features = [5,7]
new_features2 = [2,4]
new_features3 = [4,4]

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups. IDIOT!!")
    distances = []
    for group in data:
        for features in data[group]:
            #euclidian_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )
            # this is not fast. iterating through list of lists will be big O n^2. bad.
            #this is 2D only. often need N dimensions.
            euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidian_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    #i is the group. i[1] is the point nearest to i[0]
    #[:k] = subsetting list from start to k
    #this one liner above would be equivalent to
    # votes = []
    #for i in sorted(distances)[:k]
    #    votes.append(i[1])
    vote_result = Counter(votes).most_common(1)[0][0]
    print ("type(Counter(votes).most_common(1))=", type(Counter(votes).most_common(1)) )
    print ("type(Counter(votes).most_common(1)[0])=", type(Counter(votes).most_common(1)[0]) )
    print ("Counter(votes).most_common(1)=", Counter(votes).most_common(1))
    #Counter(votes).most_common(1) is a list of a tuple.
    #we only want the most common result. most_common(1)
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print ("result=", result)

[ [plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i] ] for i in dataset]
#plt.scatter(new_features[0], new_features[1], s=100, color='y', marker='s'  )
plt.scatter(new_features[0], new_features[1], s=100, color=result, marker='s'  )
#now classify second point.
result = k_nearest_neighbors(dataset, new_features2, k=3)
plt.scatter(new_features2[0], new_features2[1], s=100, color=result, marker='s'  )
result = k_nearest_neighbors(dataset, new_features3, k=3)
plt.scatter(new_features3[0], new_features3[1], s=100, color=result, marker='s'  )
new_features4 = [4,5]
result = k_nearest_neighbors(dataset, new_features4, k=3)
plt.scatter(new_features4[0], new_features4[1], s=100, color=result, marker='s'  )

plt.show()


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

'''