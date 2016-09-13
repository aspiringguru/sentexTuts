'''
Creating Our K Nearest Neighbors Algorithm - Practical Machine Learning with Python p.16
https://youtu.be/n3RqsMz3-0A?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

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
new_features = [5,7]

#for i in dataset:
#    for ii in dataset[i]:
#        plt.scatter(ii[0], ii[1], s=100, color=i)

#one liner replacement for above. [musing : is this not pythonic ?? ]
[ [plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i] ] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100, color='y', marker='s'  )
plt.show()



'''
http://matplotlib.org/api/markers_api.html

'''