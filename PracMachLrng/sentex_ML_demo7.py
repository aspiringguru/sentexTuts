'''
working exercise from sentex tutorials. with mods for clarification + api doc references.
How to program the Best Fit Line - Practical Machine Learning Tutorial with Python p.9
https://youtu.be/KLGfMGsgP34?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
linear regression model  y=mx+b
m = mean(x).mean(y) -  mean (x.y)
    ------------------------------
    (mean(x)^2 - mean(x^2)

b = mean(y) - m . mean(x)

'''

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = [1,2,3,4,5,6]
ys = [5,4,6,5,6,7]
#plt.scatter(xs, ys)
#plt.show()


xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = (mean(xs) * mean(ys) - mean(xs*ys)) / ( mean(xs)*mean(xs) - mean(xs*xs) )
    b = mean(ys) - m * mean(xs)
    return (m, b)

m,b = best_fit_slope_and_intercept(xs, ys)
#regression_line = xs*m+b
regression_line = [m*x+b for x in xs]
print ( "m={}".format(m), ", b={}".format(b) )

predict_x = 8
predict_y = (m*predict_x) + b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color = 'g', marker='s', s=50)
#plt.plot(xs, xs*m+b)
plt.plot(xs, regression_line)
plt.xlabel('xs')
plt.ylabel('ys')
plt.title("plot mx+b using linear regression fit")
plt.show()

'''
http://matplotlib.org/examples/style_sheets/plot_fivethirtyeight.html

'''