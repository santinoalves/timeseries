# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:02:57 2019

@author: Connor Simpson
"""
from netCDF4 import Dataset, num2date
import urllib.request
import pandas as pd
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
import matplotlib.pyplot as plt



url = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSDAR/Wave/IMOS_ANMN-NRS_WZ_20180731T075900Z_NRSDAR_FV01_NRSDAR-1807-SUB-Sentinel-or-Monitor-Workhorse-ADCP-16.6_END-20190213T215900Z_C-20190402T064217Z.nc'
urllib.request.urlretrieve(url, 'dataII')
data3 = Dataset('dataII')

times3 = data3.variables['TIME']
dates3 = num2date(times3[:],times3.units)
WWSH = data3.variables['WMPP']

WWSH = data3.variables['WWSH']
ts3 = pd.Series(data=WWSH[:],index=dates3)
ts3.plot()
j=[0]*2373
k=0
f=[0]*2373
for i in range(2373):
    if ts3.isnull()[i] == True:
        k=k+1
        j[k] = i
        f[i] = 1
    else:
        f[i] = 0
plt.scatter(range(2373),f)        


class missing(object):
    def __init__(self,timeseries):
        n = timeseries.size
        missing_size=0
        missing_series=[0]*n
        a=0
        for i in range(n):
            if timeseries.isnull()[i] == True:
                missing_size=missing_size+1
                missing_series[i] = 1            
            else:
                    missing_series[i] = 0
        missing_position=[0]*missing_size
        for j in range(n):
            if timeseries.isnull()[j] == True:
                missing_position[a]=j
                a=a+1
        missing_range = [] 
        start = missing_position[0]
        end = missing_position[0]
        for i in range(missing_size):
            if i>0 and missing_position[i] != missing_position[i-1]+1:
                start = missing_position[i]
                end = missing_position[i]
            elif i<missing_size-1 and missing_position[i] == missing_position[i+1]-1:
                end = missing_position[i+1]
            else:
                missing_range.append([start,end])

        self.missing_size = missing_size
        self.missing_position = missing_position
        self.missing_series = missing_series
        self.series_size = n
        self.missing_blocks = missing_range
                
            
            
            
class complete(object):
    def __init__(self,timeseries):
        n = timeseries.size
        missing_size=0
        missing_series=[0]*n
        a=0
        for i in range(n):
            if timeseries.isnull()[i] == False:
                missing_size=missing_size+1
                missing_series[i] = 1            
            else:
                    missing_series[i] = 0
        missing_position=[0]*missing_size
        for j in range(n):
            if timeseries.isnull()[j] == False:
                missing_position[a]=j
                a=a+1
        missing_range = [] 
        start = missing_position[0]
        end = missing_position[0]
        for i in range(missing_size):
            if i>0 and missing_position[i] != missing_position[i-1]+1:
                start = missing_position[i]
                end = missing_position[i]
            elif i<missing_size-1 and missing_position[i] == missing_position[i+1]-1:
                end = missing_position[i+1]
            else:
                missing_range.append([start,end])
                
        

        self.complete_size = missing_size
        self.complete_position = missing_position
        self.complete_series = missing_series
        self.series_size = n
        self.complete_blocks = missing_range
        
        
