'''
https://pythonprogramming.net/svm-in-python-machine-learning-tutorial/?completed=/svm-constraint-optimization-machine-learning-tutorial/

SVM Training - Practical Machine Learning Tutorial with Python p.26
https://youtu.be/QAs2olt7pJ4?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v


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
        self.data = data
        #{ ||w||: [w,b] }
        opt.dict = ()
        transforms = [[1,2], [-1,1], [-1,-1], [1,-1]]
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        step_sizes = [self.max_feature_value * 0.1, self.max_feature_value * 0.01, self.max_feature_value * 0.001]
        #NB: smaller steps become more expensive
        #can these steps be multithreaded?
        b_range_multiple = 5
        #this ^ is expensive.
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            #we can set optimized because convex method.
            optimized = False
            while not optimized:
                pass


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