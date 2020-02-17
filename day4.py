# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:42:42 2019

@author: Connor Simpson
"""

from netCDF4 import Dataset, num2date
import urllib.request
import pandas as pd
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima_model import ARIMA


url = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSDAR/Wave/IMOS_ANMN-NRS_WZ_20180731T075900Z_NRSDAR_FV01_NRSDAR-1807-SUB-Sentinel-or-Monitor-Workhorse-ADCP-16.6_END-20190213T215900Z_C-20190402T064217Z.nc'
urllib.request.urlretrieve(url, 'dataII')
data3 = Dataset('dataII')

times3 = data3.variables['TIME']
dates3 = num2date(times3[:],times3.units)

WMPP = data3.variables['WMPP'][:]
WWSH = data3.variables['WWSH'][:]
WSSH = data3.variables['WSSH'][:]
WPPE = data3.variables['WPPE'][:]
WPDI = data3.variables['WPDI'][:]
WWPP = data3.variables['WWPP'][:]
WWPD = data3.variables['WWPD'][:]
SWSH = data3.variables['SWSH'][:]
SWPP = data3.variables['SWPP'][:]
SWPD = data3.variables['SWPD'][:]
DEPTH = data3.variables['DEPTH'][:]
WMXH = data3.variables['WMXH'][:]
WHTH = data3.variables['WHTH'][:]
WPTH = data3.variables['WPTH'][:]
WMSH = data3.variables['WMSH'][:]
WPMH = data3.variables['WPMH'][:]
WHTE = data3.variables['WHTE'][:]
WPTE = data3.variables['WPTE'][:]
VDIR = data3.variables['VDIR'][:]


data22 = {'WMPP':WMPP,'WWSH':WWSH,'WSSH':WSSH,'WPPE':WPPE,'WPDI':WPDI,'WWPP':WWPP,'WWPD':WWPD,'SWSH':SWSH,'SWPP':SWPP,'SWPD':SWPD,'DEPTH':DEPTH,'WMXH':WMXH,'WHTH':WHTH,'WPTH':WPTH,'WMSH':WMSH,'WPMH':WPMH,'WHTE':WHTE,'WPTE':WPTE,'VDIR':VDIR,}

ts4 = pd.DataFrame(data=data22,index=dates3)

ts6 = pd.Series(WWSH[12:-1],index=dates3[12:-1])
ts6.plot()

ts6 = pd.Series(WWSH[2000:-1],index=dates3[2000:-1])


adfullertest3 = adfuller(ts6)
adfullertest3[1]
kpsstest3 = kpss(ts6)
kpsstest3[1]


ts6diff = ts6.diff()

adfullertest3 = adfuller(ts6diff[1:])
adfullertest3[1]
kpsstest3 = kpss(ts6diff[1:])
kpsstest3[1]

sm.graphics.tsa.plot_acf(ts6,lags=45)
sm.graphics.tsa.plot_pacf(ts6,lags=45)

sm.graphics.tsa.plot_acf(ts6diff[1:],lags=45)
sm.graphics.tsa.plot_pacf(ts6diff[1:],lags=45)



from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from matplotlib import pyplot


model1 = ARIMA(ts6,(12,1,12))
model = ARIMA(ts6, order=(12,1,3))
model_fit = model.fit(disp=0)
print(model_fit.summary())
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



X = ts6.values
size = int(len(X) * 0.90)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,5))
	model_fit = model.fit(disp=0,start_ar_lags=20)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
rsq = r2_score(test,predictions)
rsq
error


print('Test MSE: %.3f' % error)


plt.plot(predictions)
plt.plot(test)





from pyramid.arima import auto_arima
stepwise_model = auto_arima(X, start_p=1, start_q=1,
                           max_p=6, max_q=6, m=12,
                           start_P=0, seasonal=False,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

















