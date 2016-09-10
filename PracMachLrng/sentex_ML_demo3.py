#working exercise from sentex tutorials. with mods for clarification + api doc references.
#Regression Training and Testing - Practical Machine Learning Tutorial with Python p.4
#https://youtu.be/r4mwkS2T9aI?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
import pandas as pd
import sklearn
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

import quandl
import math
import numpy as np

stockcode = 'WIKI/GOOGL'

print ("getting data")
df = quandl.get(stockcode)
print ("type(df)={}".format(type(df)))

#['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT']= 100.0*(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']
df['PCT_change']= 100.0*(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'PCT_change']]
forecast_col = 'Adj. Close'
#replace na data with nominated value and nominate as an outlier.
df.fillna(-99999, inplace=True)
#create variable to select a fraction of number of rows in df
#will use to predict price at 'number of days out'
#math.ceil(x) : NB returns as a float.
# Return the ceiling of x as a float, the smallest integer value greater than or equal to x.
forecast_out = int(math.ceil(0.01*len(df)))
newColLabel = str(forecast_out)+" days out"
print ("type(df[forecast_col]={}".format(type(df[forecast_col])))
#type(df[forecast_col] = class 'pandas.core.series.Series'
df[newColLabel] = df[forecast_col].shift(-forecast_out)
df["% change from Adj close"] = 100.0*(df[newColLabel] - df['Adj. Close'])/df['Adj. Close']
#ie: time into the future.
df.dropna(inplace=True)
#
print ("-----with 0.1 shift-----------------------")
print ("Forecast"+newColLabel)
print (df.head())
print ("------------------------------------------")
print (list(df.dtypes.index))
df = df.drop(['% change from Adj close'], 1)
#df = df.drop([newColLabel], 1)
print (list(df.dtypes.index))
print ("newColLabel={}".format(newColLabel))
print (df.head())
#Features - in uppercase.
#labels - in lowercase
X = np.array(df.drop([newColLabel], 1))
y = np.array(df[newColLabel])
X = preprocessing.scale(X)

#X = X[:-forecast_out-1]
#subset X to exclude the dates we will be forecasting. ie exclude last forecast_out columns - 1
df.dropna(inplace=True)#don't need this really.
y = np.array(df[newColLabel])

print ("len(X)=", len(X), "len(y)=", len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
print (type(clf))
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print ("via LinearRegression method : accuracy=", accuracy, "type={}".format(type(accuracy)))

clf = svm.SVR()
print (type(clf))
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print ("via svm.SVR method : accuracy=", accuracy, "type={}".format(type(accuracy)))




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


"""