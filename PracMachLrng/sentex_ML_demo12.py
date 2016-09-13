'''
Euclidean Distance - Practical Machine Learning Tutorial with Python p.15
https://youtu.be/hl3bQySs8sM?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

https://en.wikipedia.org/wiki/Euclidean_distance
The distance (d) from p to q, or from q to p is given by the Pythagorean formula:
d(q,p) = d(p,q) = sqrt( (q1-p1)^2 + (q2-p2)^2 + .... +  (qn-pn)^2)
[recall hyptoneuse of 90 deg triangle formula h = sqrt(x^2 + y^2) where x & y are the square sides.]
euclidian distance = sqrt(Sum [i=1 to n]  (qi - pi)^2)
'''
from math import sqrt
plot1 = [1,3]
plot2 = [2,5]
euclidian_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )

print ("euclidian_distance={}".format(euclidian_distance))