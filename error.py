# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:35:34 2019

@author: csimp
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

ts6 = pd.Series(WWSH[12:200],index=dates3[12:200])
ts6.plot()

ts7 = ts6.values

def diff(ts):
    tsdiff = []
    for i in range(1,len(ts)):
        tsdiff.append(ts[i] - ts[i-1])
    return(tsdiff)

def antidiff(ts,ini):
    antidiff = [ini]
    for i in range(len(ts)):
        antidiff.append(antidiff[i]+ts[i])
    return(antidiff)

def antidiffroll(ts1,ts2):
    antidiffroll = []
    for i in range(len(ts1)):
        antidiffroll.append(ts1[i]+ts2[i])
    return(antidiffroll)

def predictAR(model,data):
    predict = []
    for j in range(len(data)-len(model.params)+1):
        predict.append(model.params[0])
        lags = []
        lags= lags + data[j:j+len(model.params)-1]
        for i in range(len(model.params)-1):
            predict[j] = predict[j] + lags[i]*model.params[len(model.params)-1-i]
    return(predict)

def mean(ts):
    mean = sum(ts)/len(ts)
    return(mean)

def rsq(pred,tru):
    SSE = 0
    for i in range(len(pred)):
        SSE=SSE + (tru[i]-pred[i])*(tru[i]-pred[i])
    SS = 0
    for i in range(len(tru)):
        SS=SS + (tru[i] - mean(tru))*(tru[i] - mean(tru))
    rsq= 1 - SSE/SS
    return(rsq)

def adjrsq(pred,tru,coef):
    rs = rsq(pred,tru)
    k = len(coef)-1
    n = len(pred)
    adjrsq = 1-(1-rs)*(n-1)/(n-k-1)
    return(adjrsq)

def rmse(pred,tru):
    from math import sqrt
    mse = 0
    for i in range(len(pred)):
        mse = mse + (pred[i]-tru[i])*(pred[i]-tru[i])/len(pred)
    rmse = sqrt(mse)
    return(rmse)
    

def relmae(pred,tru,bench):
    MAE = 0
    for i in range(len(pred)):
        MAE = MAE + abs(tru[i]-pred[i])
    MAEb = 0
    for i in range(len(bench)):
        MAEb = MAEb + abs(tru[i]-bench[i])
    relmae = MAE/MAEb
    return(relmae)
    
def predictMA(model,data):
    predictma = []
    error =[]
    for i in range(len(data)):
        error.append(data[i]-mean(data))
    for j in range(len(data)-len(model.params)+1):
        predictma.append(mean(data)+model.params[0])
        lags = []
        lags= lags + error[j:j+len(model.params)-1]
        for i in range(len(model.params)-1):
            predictma[j] = predictma[j] + lags[i]*model.params[len(model.params)-1-i]
    return(predictma)


def predictARMA(model,data):
    predictarma = []
    error =[0]*len(model.maparams)
    for j in range(len(data)-len(model.params)+1):
        error.append(0)
        predictarma.append(model.params[0])
        arlags = []
        arlags= arlags + data[j:j+len(model.arparams)]
        malags = []
        malags= malags + error[j:j+len(model.maparams)]
        lags = []
        lags= lags + data[j:j+len(model.arparams)]
        lags = lags + error[j:j+len(model.maparams)]
        for i in range(len(model.params)-1):
            predictarma[j] = predictarma[j] + (lags[i])*(model.params[len(model.params)-1-i])
        error[j] = data[j+len(model.params)-1] - predictarma[j]

    return(predictarma)


        #for i in range(len(model.arparams)):
       #     error[j] = error[j] - arlags[i]*model.arparams[len(model.arparams)-1-i]
       # for i in range(len(model.maparams)):
        #    error[j] = error[j] - malags[i]*model.maparams[len(model.maparams)-1-i]

#
size = int(len(ts7)*.8)
data = diff(ts7)
train = data[1:size]
test = data[size:]

adr = []
rmse2 = []
adr2 = []
rmse22 = []
relmae2 = []
relmae22 = []
for z in range(1,30):
    model = ARMA(train,(z,0))
    modelfit = model.fit(maxiter=100,method='css')
    coef = modelfit.params
    
    pred = predictAR(modelfit,test)
    tru = test[len(coef)-1:]
    bench = test[len(coef)-2:-1]
    
    pred2 = antidiffroll(pred,ts7[size+len(coef)-1:])
    tru2 = antidiffroll(tru,ts7[size+len(coef)-1:])
    bench2 = antidiffroll(bench,ts7[size+len(coef)-2:])
    
    adr.append(adjrsq(pred,tru,coef))
    rmse2.append(rmse(pred,tru))
    relmae2.append(relmae(pred,tru,bench))
    
    adr2.append(adjrsq(pred2,tru2,coef))
    rmse22.append(rmse(pred2,tru2))
    relmae22.append(relmae(pred2,tru2,bench2))
    print(z)

adr = []
rmse2 = []
adr2 = []
rmse22 = []
relmae2 = []
relmae22 = []
for z in range(1,30):
    model = AR(train)
    modelfit = model.fit(maxlag=z)
    coef = modelfit.params
    
    pred = predictAR(modelfit,test)
    tru = test[len(coef)-1:]
    bench = test[len(coef)-2:-1]
    
    pred2 = antidiffroll(pred,ts7[size+len(coef)-1:])
    tru2 = antidiffroll(tru,ts7[size+len(coef)-1:])
    bench2 = antidiffroll(bench,ts7[size+len(coef)-2:])
    
    adr.append(adjrsq(pred,tru,coef))
    rmse2.append(rmse(pred,tru))
    relmae2.append(relmae(pred,tru,bench))
    
    adr2.append(adjrsq(pred2,tru2,coef))
    rmse22.append(rmse(pred2,tru2))
    relmae22.append(relmae(pred2,tru2,bench2))
    print(z)


adr = []
rmse2 = []
adr2 = []
rmse22 = []
relmae2 = []
relmae22 = []
for z in range(1,5):
    model = ARMA(train,(0,z))
    modelfit = model.fit(maxiter=100,method='css')
    coef = modelfit.params
    
    pred = predictMA(modelfit,test)
    tru = test[len(coef)-1:]
    bench = test[len(coef)-2:-1]
    
    pred2 = antidiffroll(pred,ts7[size+len(coef)-1:])
    tru2 = antidiffroll(tru,ts7[size+len(coef)-1:])
    bench2 = antidiffroll(bench,ts7[size+len(coef)-2:])
    
    adr.append(adjrsq(pred,tru,coef))
    rmse2.append(rmse(pred,tru))
    relmae2.append(relmae(pred,tru,bench))
    
    adr2.append(adjrsq(pred2,tru2,coef))
    rmse22.append(rmse(pred2,tru2))
    relmae22.append(relmae(pred2,tru2,bench2))
    print(z)


 
#This part is just testing the predict functions
model = ARIMA(train,(6,0,0))
modelfit = model.fit()
a = predictAR(modelfit,train)
c2 = modelfit.predict(6,len(train))
b= []
for i in range(len(a)):
    b.append(a[i]-c2[i])

plt.plot(b)


model = AR(train)
modelfit = model.fit(maxlag=6)
a = predictAR(modelfit,train)
c = modelfit.predict(6,len(train))
b= []
for i in range(len(a)):
    b.append(a[i]-c[i])

plt.plot(c)

b= []
for i in range(len(a)):
    b.append(c2[i]-c[i])
plt.plot(b)

model = ARMA(train,(0,5))
modelfit = model.fit(tend='c')

a = predictMA(modelfit,train)
c = modelfit.predict(5,size)
b= []
for i in range(len(a)):
    b.append(a[i]-c[i])

plt.plot(b)


model = ARMA(train,(0,1))
modelfit = model.fit(tend='c')
a = predictMA(modelfit,train)
c = modelfit.predict(1,len(train))
b= []
for i in range(len(a)):
    b.append(a[i]-c[i])

plt.plot(b)


c = modelfit.predict(1,len(train))



