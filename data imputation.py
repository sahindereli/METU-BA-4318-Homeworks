import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm

#Importing data
# dataframe
df = pd.read_csv("Danish Kron.txt", sep='\t')
print(df.axes)
# on Mac you may need to use
# df = pd.read_csv("Danish Kron.txt", sep='\t', lineterminator='\r')

    

def replacezero(df):
    for row in range(df.shape[0]):
        if df.loc[row, "VALUE"] == 0.0:
            prdate=df.loc[row-1,"VALUE"]
            nxtdate=df.loc[row+1,"VALUE"]
            avg = (nxtdate+prdate) /2
            df.loc[row, "VALUE"] = avg
    return df

df=replacezero(df)

# Create slices for training and test data
size = len(df)
head = df[0:5]
tail = df [size-5:]
print("Head")
print(head)
print("Tail")
print(tail)

train = df[0:size-201]
test = df[size-200:]

#train = df.copy()
#test = df.copy()


df.DATE = pd.to_datetime(df.DATE,format="%Y-%m-%d")
df.index = df.DATE 
train.DATE = pd.to_datetime(train.DATE,format="%Y-%m-%d") 
train.index = train.DATE 
test.DATE = pd.to_datetime(train.DATE,format="%Y-%m-%d") 
test.index = test.DATE 

#Naive approach
print("Naive")
dd= np.asarray(train.VALUE)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
rms = sqrt(mean_squared_error(test.VALUE, y_hat.naive))
print("RMSE: ",rms)

#Simple average approach
print("Simple Average")
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['VALUE'].mean()
rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.avg_forecast))
print("RMSE: ",rms)

#Moving average approach
print("Moving Average")
windowsize = 15
y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train['VALUE'].rolling(windowsize).mean().iloc[-1]
rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.moving_avg_forecast))
print("RMSE: ",rms)

# Simple Exponential Smoothing
print("Simple Exponential Smoothing")
y_hat_avg = test.copy()
alpha = 0.2
fit2 = SimpleExpSmoothing(np.asarray(train['VALUE'])).fit(smoothing_level=alpha,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.SES))
print("RMSE: ",rms)

# Holt
print("Holt")
sm.tsa.seasonal_decompose(train.VALUE).plot()
result = sm.tsa.stattools.adfuller(train.VALUE)
# plt.show()

y_hat_avg = test.copy()
alpha = 0.4
fit1 = Holt(np.asarray(train['VALUE'])).fit(smoothing_level = alpha,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.Holt_linear))
print("RMSE: ",rms)

# Holt-Winters
print("Holt-Winters")
y_hat_avg = test.copy()
seasons = 10
fit1 = ExponentialSmoothing(np.asarray(train['VALUE']) ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.Holt_Winter))
print("RMSE: ",rms)

# Seasonal ARIMA
# This is a naive use of the technique. See - http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
# print("Seasonal ARIMA")
# y_hat_avg = test.copy()
# fit1 = sm.tsa.statespace.SARIMAX(train.VALUE, order=(1, 0, 0),seasonal_order=(0,1,1,1)).fit()
# y_hat_avg['SARIMA'] = fit1.predict(start="2008-12-01", end="2018-11-29", dynamic=True)
# rms = sqrt(mean_squared_error(test.VALUE, y_hat_avg.SARIMA))
# print("RMSE: ",rms)