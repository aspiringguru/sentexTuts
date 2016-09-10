#working exercise from sentex tutorials. with mods for clarification + api doc references.
#Regression Intro - Practical Machine Learning Tutorial with Python p.2
#https://youtu.be/JcI5Vnw0b2c?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
import pandas as pd
import sklearn
import quandl
import math

stockcode = 'WIKI/GOOGL'

print ("getting data")
df = quandl.get(stockcode)
#http://pandas.pydata.org/pandas-docs/stable/dsintro.html
#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

#['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT']= 100.0*(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']
df['PCT_change']= 100.0*(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'PCT_change']]
#print (list(df.dtypes.index))
#print ("after adding & selecting columns\n")
print ("-----before shift-----------------------")
print (df.head())

forecast_col = 'Adj. Close'
#replace na data with nominated value and nominate as an outlier.
#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html
#inplace : boolean, default False. If True, fill in place.
# Note: this will modify any other views on this object, (e.g. a no-copy slice for a column in a DataFrame).
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
#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.shift.html
#Series.shift(periods=1, freq=None, axis=0)
#Shift index by desired number of periods with an optional time freq
#ie: time into the future.
df.dropna(inplace=True)
#
print ("-----with 0.1 shift-----------------------")
print ("Forecast"+newColLabel)
print (df.head())
print ("------------------------------------------")
#forecast_out = int(math.ceil(0.01*len(df)))
#df['label'] = df[forecast_col].shift(-forecast_out)
#print (df.head())
