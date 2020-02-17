# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:59:14 2019

@author: Connor Simpson
"""

from netCDF4 import Dataset, num2date
import urllib.request
import pandas as pd
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

url = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/sea_surface_wave_significant_height_channel_3000/2019/NO_QAQC/IMOS_ANMN_W_20190901T000000Z_NRSDAR_FV00.nc'
urllib.request.urlretrieve(url, 'dataI')
data1 = Dataset('dataI')


url = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/sea_surface_wave_significant_height_channel_3000/2018/QAQC/IMOS_ANMN_W_20181001T000000Z_NRSDAR_FV01.nc'
urllib.request.urlretrieve(url, 'dataI')
data2 = Dataset('dataI')

VAVH = data2.variables['VAVH']
#sea_surface_wave_significant_height
#saving the data in a variable
times = data2.variables['TIME']
dates = num2date(times[:],times.units)
#Turning the timesteps into dates for the timesries

ts = pd.Series(data=VAVH[:],index=dates)
#producing a pandas timeseries

ts.plot()
ts.hist()
#Appears fairly normal probably stationary
adfullertest0 = adfuller(ts[1:])
adfullertest0[1]
kpsstest = kpss(ts)
kpsstest[1]
#from the kpss test this is not stationary 
sm.graphics.tsa.plot_acf(ts)
sm.graphics.tsa.plot_pacf(ts)

tsdiff2 = ts.diff()
tsdiff2[1:].plot()
tsdiff2[1:].hist()
#looks much better

adfullertest = adfuller(tsdiff2[1:])
adfullertest[1]
kpsstestdiff2 = kpss(tsdiff2[1:])
kpsstestdiff2[1]
#the test confirms this is significant
sm.graphics.tsa.plot_acf(tsdiff2[1:],lags=45)
sm.graphics.tsa.plot_pacf(tsdiff2[1:],lags=45)


url = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/REAL_TIME/NRSDAR/sea_surface_wave_significant_height_channel_3000/2019/QAQC/IMOS_ANMN_W_20190301T000000Z_NRSDAR_FV01.nc'
urllib.request.urlretrieve(url, 'dataI')
data1 = Dataset('dataI')
times1 = data1.variables['TIME']
dates1 = num2date(times1[:],times.units)
VAVH1 = data1.variables['VAVH']
ts1 = pd.Series(data=VAVH1[:],index=dates1)
ts1.plot()
ts1.hist()
#might not be stationary
kpsstest1 = kpss(ts1)
kpsstest1[1]
#kpss sujests stationary 
sm.graphics.tsa.plot_acf(ts1)
sm.graphics.tsa.plot_pacf(ts1)


tsdiff = ts1.diff()
tsdiff[1:].plot()
tsdiff[1:].hist()
#looks much better
kpsstestdiff = kpss(tsdiff[1:])
kpsstestdiff[1]
#the test confirms this is significant
sm.graphics.tsa.plot_acf(tsdiff[1:])
sm.graphics.tsa.plot_pacf(tsdiff[1:])


url = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSDAR/Wave/IMOS_ANMN-NRS_WZ_20180731T075900Z_NRSDAR_FV01_NRSDAR-1807-SUB-Sentinel-or-Monitor-Workhorse-ADCP-16.6_END-20190213T215900Z_C-20190402T064217Z.nc'
urllib.request.urlretrieve(url, 'dataII')
data3 = Dataset('dataII')

times3 = data3.variables['TIME']
dates3 = num2date(times3[:],times.units)
WWSH = data3.variables['WWSH']
ts3 = pd.Series(data=WWSH[:],index=dates3)
ts3.plot()
adfullertest3 = adfuller(ts3)
adfullertest3[1]
kpsstest3 = kpss(ts3)
kpsstest3[1]

ts3.size
ts3.index

