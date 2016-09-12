'''
working exercise from sentex tutorials. with mods for clarification + api doc references.
Classification w/ K Nearest Neighbors Intro - Practical Machine Learning Tutorial with Python p.13
https://youtu.be/44jq6ano5n0?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

linear regression model  y=mx+b
m = mean(x).mean(y) -  mean (x.y)
    ------------------------------
    (mean(x)^2 - mean(x^2)

b = mean(y) - m . mean(x)

Coefficient of Linear Regression, ranges from 0 to 1.
An R2 of 0 means that the dependent variable cannot be predicted from the independent variable.
No correlation:  If there is no linear correlation or a weak linear correlation, r is
     close to 0.  A value near zero means that there is a random, nonlinear relationship
     between the two variables
An R2 of 1 means the dependent variable can be predicted without error from the independent variable.
A perfect correlation of +/- 1 occurs only when the data points all lie exactly on a
     straight line.  If r = +1, the slope of this line is positive.  If r = -1, the slope of this
     line is negative.

An R2 between 0 and 1 indicates the extent to which the dependent variable is predictable.
An R2 of 0.10 means that 10 percent of the variance in Y is predictable from X;
an R2 of 0.20 means that 20 percent is predictable; and so on.
The other % of error is unexplained by the model.
A correlation greater than 0.8 is generally described as strong, whereas a correlation less than 0.5 is
generally described as weak.  These values can vary based upon the "type" of data being examined.
A study utilizing scientific data may require a stronger correlation than a study using social science data.

r^2 = 1 -[ (Standard Error y^) / (standard Error (mean of y) ]
'''

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = [1,2,3,4,5,6]
ys = [5,4,6,5,6,7]
#plt.scatter(xs, ys)
#plt.show()


def create_dataset(hm, variance, step=2, correlation=False):
    #correlation to be +ive/-ive or none.
    # if correlation is 'pos' then positive correlation
    # if correlation is 'neg'  then negative correlation
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

        #so far we have generated an array of values within range (1-variance) to (1+variance)
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = (mean(xs) * mean(ys) - mean(xs*ys)) / ( mean(xs)*mean(xs) - mean(xs*xs) )
    b = mean(ys) - m * mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    #error between the predicted y line and the actual points.
    return sum((ys_line-ys_orig)**2)
def coefficient_of_determination(ys_orig, ys_line):
    #
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    #creat array filled with mean of original y values.
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1- (squared_error_regr/squared_error_y_mean)

#create data using new function
#create_dataset(hm, variance, step=2, correlation=False),
# returns np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
xs, ys = create_dataset(40, 5, 2, correlation='pos')
print ("xs=", xs)
print ("ys=", ys)

m,b = best_fit_slope_and_intercept(xs, ys)
#regression_line = xs*m+b
regression_line = [m*x+b for x in xs]
print ( "m={}".format(m), ", b={}".format(b) )

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print ("r_squared=", r_squared)

plt.scatter(xs, ys)
#plt.plot(xs, xs*m+b)
plt.plot(xs, regression_line)
plt.scatter(predict_x, predict_y, color = 'g', marker='s', s=100)
plt.xlabel('xs')
plt.ylabel('ys')
plt.title("plot mx+b using linear regression fit")
plt.show()

'''
http://matplotlib.org/examples/style_sheets/plot_fivethirtyeight.html

'''