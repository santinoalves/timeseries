# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:09:15 2019

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
ts4.plot()
ts4[['WSSH','DEPTH']].plot()

import numpy as np
missingValues = np.multiply(ts4[ts4.columns[:]].isna(),1)
missingTS = pd.DataFrame(missingValues,dates3)
missingTS['VDIR'].plot()

missing_percent = []
missing_percent2 = []
missing_percent3 = {}
for i in range(len(ts4.columns)):
    missing_percent.append([ts4.columns[i],sum(missingValues[ts4.columns[i]][:])/len(missingValues[ts4.columns[i]])])
    missing_percent2.append(sum(missingValues[ts4.columns[i]][:])/len(missingValues[ts4.columns[i]]))
    missing_percent3[ts4.columns[i]] = sum(missingValues[ts4.columns[i]][:])/len(missingValues[ts4.columns[i]])
 
    

aa = pd.DataFrame(data= missing_percent2,index = ts4.columns)


from numpy import *

    
class missing(object):
    def __init__(self,missingValues):
        missing_range = []
        complete_range = []
        missing_dates = []
        complete_dates = []
        complete_block_length = []
        missing_block_length  = []
        start1 = 0
        end1 = 0
        start = 0
        end = 0
        for i in range(len(missingValues)):
            if i>0 and missingValues[i] == 1 and  missingValues[i-1] != 1:
                start = i
                if missingValues[i+1] != 1:
                    end = i
                    missing_range.append([start,end])
                    missing_dates.append([missingValues.index[start],missingValues.index[end]])
                    missing_block_length.append(end-start+1)
            elif i<len(missingValues)-1 and missingValues[i] == 1 and missingValues[i+1] !=1 :
                end = i
                missing_range.append([start,end])
                missing_dates.append([missingValues.index[start],missingValues.index[end]])
                missing_block_length.append(end-start+1)
            elif i == len(missingValues)-1 and missingValues[i] == 1:
                end = i
                missing_range.append([start,end])
                missing_dates.append([missingValues.index[start],missingValues.index[end]])
                missing_block_length.append(end-start+1)
            else:
                pass
        for i in range(len(missingValues)):
            if i>0 and missingValues[i] == 0 and  missingValues[i-1] != 0:
                start1 = i
                if missingValues[i+1] != 0:
                    end1 = i
                    complete_range.append([start1,end1])
                    complete_dates.append([missingValues.index[start1],missingValues.index[end1]])
                    complete_block_length.append(end1-start1+1)
            elif i<len(missingValues)-1 and missingValues[i] == 0 and missingValues[i+1] !=0 :
                end1 = i
                complete_range.append([start1,end1])
                complete_dates.append([missingValues.index[start1],missingValues.index[end1]])
                complete_block_length.append(end1-start1+1)
            elif i == len(missingValues)-1 and missingValues[i] == 0:
                end1 = i
                complete_range.append([start1,end1])
                complete_dates.append([missingValues.index[start1],missingValues.index[end1]])
                complete_block_length.append(end1-start1+1)
            else:
                pass
        self.missing_blocks = missing_range
        self.complete_blocks = complete_range
        self.complen = complete_block_length
        self.misslen = missing_block_length
        self.compdates = complete_dates
        self.missdates = missing_dates



from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

adfullertest3 = adfuller(ts4['WWSH'][12:-1])
adfullertest3[1]
kpsstest3 = kpss(ts4['WWSH'][12:-1])
kpsstest3[1]



times5 = data3.variables['TIME'][12:-1]
dates5 = num2date(times5[:],times3.units)
WWSH5 = data3.variables['WWSH'][12:-1]
ts5 = pd.Series(data=WWSH5,index=dates5)

ts5diff = ts5.diff()

adfullertest3 = adfuller(ts5diff[1:])
adfullertest3[1]
kpsstest3 = kpss(ts5diff[1:])
kpsstest3[1]























