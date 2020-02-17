# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:08:14 2020

@author: Connor Simpson
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

WWSHTestdata = pd.read_csv("WWSHTestdata.csv")
WWSHTraindata = pd.read_csv("WWSHTraindata.csv")

WWSHTraindataxwind = pd.read_csv("WWSHTraindataxwind.csv")
WWSHTraindataxcndc = pd.read_csv("WWSHTraindataxcndc.csv")
WWSHTraindataxdartemp = pd.read_csv("WWSHTraindataxdartemp.csv")
WWSHTraindataxbeagtemp = pd.read_csv("WWSHTraindataxbeagtemp.csv")

WWSHtestdataxwind = pd.read_csv("WWSHtestdataxwind.csv")
WWSHtestdataxcndc = pd.read_csv("WWSHtestdataxcndc.csv")
WWSHtestdataxdartemp = pd.read_csv("WWSHtestdataxdartemp.csv")
WWSHtestdataxbeagtemp = pd.read_csv("WWSHtestdataxbeagtemp.csv")


WWSHtestdataxwind = pd.read_csv("WWSHtestdataxwind.csv")
WWSHtestdataxcndc = pd.read_csv("WWSHtestdataxcndc.csv")
WWSHtestdataxdartemp = pd.read_csv("WWSHtestdataxdartemp.csv")
WWSHtestdataxbeagtemp = pd.read_csv("WWSHtestdataxbeagtemp.csv")

#####

WWSHTestdata = WWSHTestdata.iloc[:,1:]
WWSHTraindata = WWSHTraindata.iloc[:,1:]

WWSHTraindataxwind = WWSHTraindataxwind.iloc[:,1:]
WWSHTraindataxcndc = WWSHTraindataxcndc.iloc[:,1:36]
WWSHTraindataxdartemp = WWSHTraindataxdartemp.iloc[:,1:36]
WWSHTraindataxbeagtemp = WWSHTraindataxbeagtemp.iloc[:,1:36]

WWSHtestdataxwind = WWSHtestdataxwind.iloc[:,1:]
WWSHtestdataxcndc = WWSHtestdataxcndc.iloc[:,1:36]
WWSHtestdataxdartemp = WWSHtestdataxdartemp.iloc[:,1:36]
WWSHtestdataxbeagtemp = WWSHtestdataxbeagtemp.iloc[:,1:36]





regressorWWSHxWind = MLPRegressor()
regressor = MLPRegressor.fit(regressor,bioChemAll3["darwinWave_WWSH"][:,1:],bioChemAll3["darwinBiochem_TEMP"])
regressor.fit(bioChemAll3["darwinWave_WWSH"],bioChemAll3["darwinBiochem_TEMP"])

regressor.fit([biochemall5["darwinBiochem_TEMP"].values,biochemall5["beagleBiochem_TEMP"]],np.array([biochemall5["darwinWave_WWSH"],biochemall5["darwinWave_WWSH"]]))


WWSHxWindtrain = pd.concat([WWSHTraindata,WWSHTraindataxwind],axis=1)
WWSHxWindtest = pd.concat([WWSHTestdata,WWSHtestdataxwind],axis=1)
pred = []
for i in range(0,5):
    
    WWSHdata = WWSHxWindtrain.iloc[:,[i,i*4+5,i*4+6,i*4+7,i*4+8]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    WINDtrain = WWSHdata.iloc[:,[1,2,3,4]]
    
    regressorWWSHxWind = MLPRegressor()
    regressorWWSHxWind.fit(WINDtrain,WWSHtrain)
    
    WWSHdatatest = WWSHxWindtest.iloc[:,[i,i*4+5,i*4+6,i*4+7,i*4+8]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHtest = WWSHdatatest.iloc[:,0]
    WINDtest = WWSHdatatest.iloc[:,[1,2,3,4]]
    
    pred = regressorWWSHxWind.predict(WINDtest)
    plt.plot(pred)
    
    
WWSHxtemptrain = pd.concat([WWSHTraindata,WWSHTraindataxdartemp],axis=1)
WWSHxtemptest = pd.concat([WWSHTestdata,WWSHtestdataxdartemp],axis=1)
pred = []
for i in range(0,5):
    
    WWSHdata = WWSHxtemptrain.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    temptrain = WWSHdata.iloc[:,[1,2,3,4,5,6,7]]
    
    regressorWWSHxtemp = MLPRegressor()
    regressorWWSHxtemp.fit(temptrain,WWSHtrain)
    
    WWSHdatatest = WWSHxtemptest.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHtest = WWSHdatatest.iloc[:,0]
    temptest = WWSHdatatest.iloc[:,[1,2,3,4,5,6,7]]
    
    pred = regressorWWSHxtemp.predict(temptest)
    plt.plot(pred)   
    

WWSHxtemptrain = pd.concat([WWSHTraindata,WWSHTraindataxdartemp,WWSHTraindataxwind],axis=1)
WWSHxtemptest = pd.concat([WWSHTestdata,WWSHtestdataxdartemp,WWSHtestdataxwind],axis=1)
pred = []
for i in range(0,5):
    
    WWSHdata = WWSHxtemptrain.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11,i*4+40,i*4+41,i*4+42,i*4+43]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    temptrain = WWSHdata.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]
    
    regressorWWSHxtemp = MLPRegressor()
    regressorWWSHxtemp.fit(temptrain,WWSHtrain)
    
    WWSHdatatest = WWSHxtemptest.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11,i*4+40,i*4+41,i*4+42,i*4+43]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHtest = WWSHdatatest.iloc[:,0]
    temptest = WWSHdatatest.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]
    
    pred = regressorWWSHxtemp.predict(temptest)
    plt.plot(pred)   
WWaa = WWSHtest.reset_index()
plt.plot(pred) 
plt.plot(WWaa.iloc[:,1])   
    
WWSHxtemptrain = pd.concat([WWSHTraindata,WWSHTraindataxdartemp,WWSHTraindataxwind,WWSHTraindataxcndc],axis=1)
WWSHxtemptest = pd.concat([WWSHTestdata,WWSHtestdataxdartemp,WWSHtestdataxwind,WWSHtestdataxcndc],axis=1)
for i in range(0,5):
    WWSHdata = WWSHxtemptrain.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11,i*4+40,i*4+41,i*4+42,i*4+43,i*7+60,i*7+61,i*7+62,i*7+63,i*7+64,i*7+65,i*7+66]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    temptrain = WWSHdata.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
    
    regressorWWSHxtemp = MLPRegressor()
    regressorWWSHxtemp.fit(temptrain,WWSHtrain)
    
    WWSHdatatest = WWSHxtemptest.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11,i*4+40,i*4+41,i*4+42,i*4+43,i*7+60,i*7+61,i*7+62,i*7+63,i*7+64,i*7+65,i*7+66]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHdatatest = WWSHdatatest.reset_index().iloc[:,1:20]
    globals()['WWSHtest'+str(i)]  = WWSHdatatest.iloc[:,0]
    temptest = WWSHdatatest.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
    globals()['pred'+str(i)] = regressorWWSHxtemp.predict(temptest)
    
WWaa = WWSHtest.reset_index()
plt.plot(pred1) 
plt.plot(WWSHtest1)   


WWSHxtemptrain = pd.concat([WWSHTraindata,WWSHTraindataxdartemp,WWSHTraindataxwind,WWSHTraindataxcndc,WWSHTraindataxbeagtemp],axis=1)
WWSHxtemptest = pd.concat([WWSHTestdata,WWSHtestdataxdartemp,WWSHtestdataxwind,WWSHtestdataxcndc,WWSHtestdataxbeagtemp],axis=1)
for i in range(0,5):
    WWSHdata = WWSHxtemptrain.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11,i*4+40,i*4+41,i*4+42,i*4+43,i*7+60,i*7+61,i*7+62,i*7+63,i*7+64,i*7+65,i*7+66,i*7+95,i*7+96,i*7+97,i*7+98,i*7+99,i*7+100,i*7+101]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    temptrain = WWSHdata.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]
    
    regressorWWSHxtemp = MLPRegressor()
    regressorWWSHxtemp.fit(temptrain,WWSHtrain)
    
    WWSHdatatest = WWSHxtemptest.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11,i*4+40,i*4+41,i*4+42,i*4+43,i*7+60,i*7+61,i*7+62,i*7+63,i*7+64,i*7+65,i*7+66,i*7+95,i*7+96,i*7+97,i*7+98,i*7+99,i*7+100,i*7+101]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHdatatest = WWSHdatatest.reset_index().iloc[:,1:27]
    globals()['WWSHtest'+str(i)]  = WWSHdatatest.iloc[:,0]
    temptest = WWSHdatatest.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]
    globals()['pred'+str(i)] = regressorWWSHxtemp.predict(temptest)
    
WWaa = WWSHtest.reset_index()
plt.plot(pred4) 
plt.plot(WWSHtest4)   






















df = pd.DataFrame([1,2,3])
df2 = pd.DataFrame([1,2,3,4])
df3 = pd.concat([df,df2],axis=1)
df5= pd.DataFrame()
df5.append(df)
df5.append(df2)
df5.aggregate(df)
df.











WWSHxWind = 2
regressorWWSHxWind = MLPWWSHdataRegressor()
regressorWWSHxWind.fit()


