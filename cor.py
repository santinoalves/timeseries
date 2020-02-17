# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:22:32 2019

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
from statsmodels.tsa.arima_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import pearsonr
import adjustTimeSeries as ats
url5 = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/QLD/DARBGF/Biogeochem_timeseries/IMOS_ANMN-QLD_CFKSTUZ_20130621T054453Z_DARBGF_FV01_DARBGF-1306-SBE-16plus-29.9_END-20140109T054453Z_C-20150401T060718Z.nc'

urllib.request.urlretrieve(url5, 'dataIIIII')
data5 = Dataset('dataIIIII')


bgBiochemicalSB = [
'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/QLD/DARBGF/Biogeochem_timeseries/IMOS_ANMN-QLD_CFKSTUZ_20130621T054453Z_DARBGF_FV01_DARBGF-1306-SBE-16plus-29.9_END-20140109T054453Z_C-20150401T060718Z.nc',
'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSDAR/Biogeochem_timeseries/IMOS_ANMN-NRS_CFKSTUZ_20130621T064645Z_NRSDAR_FV01_NRSDAR-1306-SUB-SBE16plus-22_END-20140119T011452Z_C-20170829T014429Z.nc'
]


biochem = ats.readTimeSeries(['TEMP','CNDC','PRES_REL','TURB','PAR','CHLF','PSAL','DEPTH'],bgBiochemicalSB,'bgBiochemicalSB','TEMP_quality_control')






