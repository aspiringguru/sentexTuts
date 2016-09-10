#working exercise from sentex tutorials. with mods for clarification + api doc references.
#Regression Intro - Practical Machine Learning Tutorial with Python p.2
#https://youtu.be/JcI5Vnw0b2c?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
import pandas as pd
import sklearn
import quandl

stockcode = 'WIKI/GOOGL'

print ("getting data")
df = quandl.get(stockcode)
print (type(df))
#http://pandas.pydata.org/pandas-docs/stable/dsintro.html
#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
print (df.dtypes.index)
print (list(df.dtypes.index))
print (type(df.dtypes.index))
print (df.head())

#['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT']= 100.0*(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']
df['PCT_change']= 100.0*(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'PCT_change']]
print (list(df.dtypes.index))
print ("after adding & selecting columns\n")
print (df.head())

