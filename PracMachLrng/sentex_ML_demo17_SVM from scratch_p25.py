'''
https://youtu.be/AbVtcUBlBok?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
https://pythonprogramming.net/svm-in-python-machine-learning-tutorial/?completed=/svm-constraint-optimization-machine-learning-tutorial/
Creating an SVM from scratch - Practical Machine Learning Tutorial with Python p.25

'''

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

   # train (scikitlearn uses a fit method to train model.
    def fit(self, data):
        pass

    def predict(self,features):
        # sign( x.w+b )   {identify if +ive or -ive}
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)

        return classification

data_dict = {-1:np.array([[1,7], [2,8], [3,8],]),
             1:np.array([[5,1], [6,-1], [7,3],])}



"""
https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
numpy.dot(a, b, out=None) = Dot product of two arrays.

http://scikit-learn.org/stable/modules/svm.html
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

textbook, lecture notes and MOOC course
https://web.stanford.edu/~boyd/cvxbook/
https://lagunita.stanford.edu/courses/Engineering/CVX101/Winter2014/about

"""