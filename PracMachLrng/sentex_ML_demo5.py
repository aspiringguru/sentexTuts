#working exercise from sentex tutorials. with mods for clarification + api doc references.
#Pickling and Scaling - Practical Machine Learning Tutorial with Python p.6
#https://youtu.be/za5s7RB_VLw?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
#update from lecture Testing Assumptions - Practical Machine Learning Tutorial with Python p.12
#https://youtu.be/Kpxwl2u-Wgk?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
'''
linear regression model  y=mx+b
'''

import pandas as pd
import sklearn
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

import quandl
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

stockcode = 'WIKI/GOOGL'

print ("setting api key")
quandl.ApiConfig.api_key = 'WdAEsUYpx6RRBRDqRoDH'
print ("getting data")
df = quandl.get(stockcode)
print ("type(df)={}".format(type(df)))#class = 'pandas.core.frame.DataFrame'

#['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT']= 100.0*(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']
df['PCT_change']= 100.0*(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'PCT_change']]
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

print ("df.head()=\n", df.head())

forecast_out = int(math.ceil(0.1*len(df)))
newColLabel = str(forecast_out)+" days out"
print ("forecast_out={}".format(forecast_out))

df[newColLabel] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop([newColLabel], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
#subset X from last 'forecast_out' days to end.
X = X[:-forecast_out]
#subset X from start to exclude last 'forecast_out' days
#NB: earlier versions had error here sectioning the datafram.
#graph plotted from earlier bad code was obviously disjointed.

df.dropna(inplace=True)
y = np.array(df[newColLabel])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

'''
clf = LinearRegression(n_jobs=-1)#If -1 all CPUs are use
#NB: classifier is untrained at this point.
clf.fit(X_train, y_train)
#this is the slow tedious step that takes a lot of time.
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
    #https://docs.python.org/3/library/pickle.html
    #pickle.dump(obj, file, protocol=None, *, fix_imports=True)
    #Write a pickled representation of obj to the open file object file.
'''
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
#pickle.load(file, *, fix_imports=True, encoding="ASCII", errors="strict")
# Read a pickled object representation from the open file object file and
# return the reconstituted object hierarchy specified therein.


accuracy = clf.score(X_test, y_test)
print ("via LinearRegression method : accuracy=", accuracy, "type={}".format(type(accuracy)))

forecast_set = clf.predict(X_lately)

print ("forecast_set : \n{}".format(forecast_set))
print ("forecast_out : {}".format(forecast_out))

print (list(df.dtypes.index))
print (type(df.dtypes.index))

df['Forecast'] = np.nan
#generate new column filled with 'not a number'.

last_date = df.iloc[-1].name
#gets last row/column? of array
#http://pandas.pydata.org/pandas-docs/stable/indexing.html
#.iloc is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
# .iloc will raise IndexError if a requested indexer is out-of-bounds, except slice indexers which allow
# out-of-bounds indexing. (this conforms with python/numpy slice semantics).
print ("type(last_date)=", type(last_date))
last_unix = last_date.timestamp()#class = 'pandas.tslib.Timestamp'
print ("last_unix=", last_unix, ", type(last_unix)", type(last_unix))
one_day = 86400 #60x60x24=86400
next_unix = last_unix + one_day

print ("len(df.columns)=", len(df.columns), "type(len(df.columns))=", type(len(df.columns)))

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print ("df.tail()=\n", df.tail(len(df.columns)+2))
print ("df.head()=\n", df.head())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(stockcode)
plt.show()
#plt.savefig('foo.png', bbox_inches='tight')
"""
api doc references below
http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.drop.html
http://scikit-learn.org/stable/modules/preprocessing.html
http://scikit-learn.org/stable/modules/cross_validation.html
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm
svm = support vector machines
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
http://pandas.pydata.org/pandas-docs/stable/dsintro.html
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html
inplace : boolean, default False. If True, fill in place.
 Note: this will modify any other views on this object, (e.g. a no-copy slice for a column in a DataFrame).
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.shift.html
Series.shift(periods=1, freq=None, axis=0)
Shift index by desired number of periods with an optional time freq

http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html
Standardize a dataset along any axis, Center to the mean and component wise scale to unit variance.
NB: refer back to preparing data for statistics Z tables.

http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html
sklearn.cross_validation.train_test_split(*arrays, **options)
Split arrays or matrices into random train and test subsets
test_size : float, int, or None (default is None)
If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
If int, represents the absolute number of test samples.
If None, the value is automatically set to the complement of the train size.
If train size is also None, test size is set to 0.25.
returns : splitting : list, length = 2 * len(arrays),
List containing train-test split of inputs.


http://scikit-learn.org/stable/modules/cross_validation.htm


http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
Support vector machines.
potentially less accurate.

can the algorythm be threaed. look for n_jobs
ie : LinearRegression
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
n_jobs : int, optional, default 1
The number of jobs to use for the computation. If -1 all CPUs are used.
This will only provide speedup for n_targets > 1 and sufficient large problems.

http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale
sklearn.preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True)
Center to the mean and component wise scale to unit variance.

https://docs.python.org/3/library/pickle.html


"""