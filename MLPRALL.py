# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:14:01 2020

@author: Connor Simpson
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


WWSHTestdata2 = pd.read_csv("WWSHTestdata.csv")
WWSHTraindata2 = pd.read_csv("WWSHTraindata.csv")

WWSHTraindataxwind2 = pd.read_csv("WWSHTraindataxwind.csv")
WWSHTraindataxcndc2 = pd.read_csv("WWSHTraindataxcndc.csv")
WWSHTraindataxdartemp2 = pd.read_csv("WWSHTraindataxdartemp.csv")
WWSHTraindataxpar2 = pd.read_csv("WWSHTraindataxpar.csv")
WWSHTraindataxturb2 = pd.read_csv("WWSHTraindataxturb.csv")
WWSHTraindataxcphl2 = pd.read_csv("WWSHTraindataxcphl.csv")



WWSHtestdataxwind2 = pd.read_csv("WWSHtestdataxwind.csv")
WWSHtestdataxcndc2 = pd.read_csv("WWSHtestdataxcndc.csv")
WWSHtestdataxdartemp2 = pd.read_csv("WWSHtestdataxdartemp.csv")
WWSHtestdataxpar2 = pd.read_csv("WWSHtestdataxpar.csv")
WWSHtestdataxturb2 = pd.read_csv("WWSHtestdataxturb.csv")
WWSHtestdataxcphl2 = pd.read_csv("WWSHtestdataxcphl.csv")

#################################################

WWSHTestdata2 = WWSHTestdata2.iloc[:,1:]
WWSHTraindata2 = WWSHTraindata2.iloc[:,1:]

WWSHTraindataxwind2 = WWSHTraindataxwind2.iloc[:,1:]
WWSHTraindataxcndc2 = WWSHTraindataxcndc2.iloc[:,1:36]
WWSHTraindataxdartemp2 = WWSHTraindataxdartemp2.iloc[:,1:36]
WWSHTraindataxpar2 = WWSHTraindataxpar2.iloc[:,1:]
WWSHTraindataxturb2 = WWSHTraindataxturb2.iloc[:,1:36]
WWSHTraindataxcphl2 = WWSHTraindataxcphl2.iloc[:,1:36]

WWSHtestdataxwind2 = WWSHtestdataxwind2.iloc[:,1:]
WWSHtestdataxcndc2 = WWSHtestdataxcndc2.iloc[:,1:36]
WWSHtestdataxdartemp2 = WWSHtestdataxdartemp2.iloc[:,1:36]
WWSHtestdataxpar2 = WWSHtestdataxpar2.iloc[:,1:]
WWSHtestdataxturb2 = WWSHtestdataxturb2.iloc[:,1:36]
WWSHtestdataxcphl2 = WWSHtestdataxcphl2.iloc[:,1:36]

#################################################



WWSHxtemptrain = pd.concat([WWSHTraindata2,WWSHTraindataxdartemp2,WWSHTraindataxwind2,WWSHTraindataxcndc2],axis=1)
WWSHxtemptest = pd.concat([WWSHTestdata2,WWSHtestdataxdartemp2,WWSHtestdataxwind2,WWSHtestdataxcndc2],axis=1)
for i in range(0,5):
    WWSHdata = WWSHxtemptrain.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11,i*4+40,i*4+41,i*4+42,i*4+43,i*7+60,i*7+61,i*7+62,i*7+63,i*7+64,i*7+65,i*7+66]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    temptrain = WWSHdata.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
    
    regressorWWSHxtemp = MLPRegressor()
    regressorWWSHxtemp.fit(temptrain,WWSHtrain)
    
    WWSHdatatest = WWSHxtemptest.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11,i*4+40,i*4+41,i*4+42,i*4+43,i*7+60,i*7+61,i*7+62,i*7+63,i*7+64,i*7+65,i*7+66]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHdatatest = WWSHdatatest.reset_index().iloc[:,1:27]
    globals()['WWSHtest'+str(i)]  = WWSHdatatest.iloc[:,0]
    temptest = WWSHdatatest.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
    globals()['pred'+str(i)] = regressorWWSHxtemp.predict(temptest)


pred0 = pd.DataFrame(pred0)
pred0.to_csv("MLPRWWSHAll0.csv")
pred1 = pd.DataFrame(pred1)
pred1.to_csv("MLPRWWSHAll1.csv")
pred2 = pd.DataFrame(pred2)
pred2.to_csv("MLPRWWSHAll2.csv")
pred3 = pd.DataFrame(pred3)
pred3.to_csv("MLPRWWSHAll3.csv")
pred4 = pd.DataFrame(pred4)
pred4.to_csv("MLPRWWSHAll4.csv")

WWSHtest0 = pd.DataFrame(WWSHtest0)
WWSHtest0.to_csv("MLPRWWSHAll0tru.csv")
WWSHtest1 = pd.DataFrame(WWSHtest1)
WWSHtest1.to_csv("MLPRWWSHAll1tru.csv")
WWSHtest2 = pd.DataFrame(WWSHtest2)
WWSHtest2.to_csv("MLPRWWSHAll2tru.csv")
WWSHtest3 = pd.DataFrame(WWSHtest3)
WWSHtest3.to_csv("MLPRWWSHAll3tru.csv")
WWSHtest4 = pd.DataFrame(WWSHtest4)
WWSHtest4.to_csv("MLPRWWSHAll4tru.csv")








WWSHxtemptrain = pd.concat([WWSHTraindata2,WWSHTraindataxwind2],axis=1)
WWSHxtemptest = pd.concat([WWSHTestdata2,WWSHtestdataxwind2],axis=1)
for i in range(0,5):
    WWSHdata = WWSHxtemptrain.iloc[:,[i,i*4+5,i*4+6,i*4+7,i*4+8]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    temptrain = WWSHdata.iloc[:,[1,2,3,4]]
    
    regressorWWSHxtemp = MLPRegressor()
    regressorWWSHxtemp.fit(temptrain,WWSHtrain)
    
    WWSHdatatest = WWSHxtemptest.iloc[:,[i,i*4+5,i*4+6,i*4+7,i*4+8]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHdatatest = WWSHdatatest.reset_index().iloc[:,1:27]
    globals()['WWSHtestWind'+str(i)]  = WWSHdatatest.iloc[:,0]
    temptest = WWSHdatatest.iloc[:,[1,2,3,4]]
    globals()['predWind'+str(i)] = regressorWWSHxtemp.predict(temptest)


predWind0 = pd.DataFrame(predWind0)
predWind0.to_csv("MLPRWWSHAllWind0.csv")
predWind1 = pd.DataFrame(predWind1)
predWind1.to_csv("MLPRWWSHAllWind1.csv")
predWind2 = pd.DataFrame(predWind2)
predWind2.to_csv("MLPRWWSHAllWind2.csv")
predWind3 = pd.DataFrame(predWind3)
predWind3.to_csv("MLPRWWSHAllWind3.csv")
predWind4 = pd.DataFrame(predWind4)
predWind4.to_csv("MLPRWWSHAllWind4.csv")

WWSHtestWind0 = pd.DataFrame(WWSHtestWind0)
WWSHtestWind0.to_csv("MLPRWWSHAllWind0tru.csv")
WWSHtestWind1 = pd.DataFrame(WWSHtestWind1)
WWSHtestWind1.to_csv("MLPRWWSHAllWind1tru.csv")
WWSHtestWind2 = pd.DataFrame(WWSHtestWind2)
WWSHtestWind2.to_csv("MLPRWWSHAllWind2tru.csv")
WWSHtestWind3 = pd.DataFrame(WWSHtestWind3)
WWSHtestWind3.to_csv("MLPRWWSHAllWind3tru.csv")
WWSHtestWind4 = pd.DataFrame(WWSHtestWind4)
WWSHtestWind4.to_csv("MLPRWWSHAllWind4tru.csv")


WWSHxtemptrain = pd.concat([WWSHTraindata2,WWSHTraindataxcndc2],axis=1)
WWSHxtemptest = pd.concat([WWSHTestdata2,WWSHtestdataxcndc2],axis=1)
for i in range(0,5):
    WWSHdata = WWSHxtemptrain.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    temptrain = WWSHdata.iloc[:,[1,2,3,4,5,6,7]]
    
    regressorWWSHxtemp = MLPRegressor()
    regressorWWSHxtemp.fit(temptrain,WWSHtrain)
    
    WWSHdatatest = WWSHxtemptest.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHdatatest = WWSHdatatest.reset_index().iloc[:,1:27]
    globals()['WWSHtestcndc'+str(i)]  = WWSHdatatest.iloc[:,0]
    temptest = WWSHdatatest.iloc[:,[1,2,3,4,5,6,7]]
    globals()['predcndc'+str(i)] = regressorWWSHxtemp.predict(temptest)


predcndc0 = pd.DataFrame(predcndc0)
predcndc0.to_csv("MLPRWWSHAllcndc0.csv")
predcndc1 = pd.DataFrame(predcndc1)
predcndc1.to_csv("MLPRWWSHAllcndc1.csv")
predcndc2 = pd.DataFrame(predcndc2)
predcndc2.to_csv("MLPRWWSHAllcndc2.csv")
predcndc3 = pd.DataFrame(predcndc3)
predcndc3.to_csv("MLPRWWSHAllcndc3.csv")
predcndc4 = pd.DataFrame(predcndc4)
predcndc4.to_csv("MLPRWWSHAllcndc4.csv")

WWSHtestcndc0 = pd.DataFrame(WWSHtestcndc0)
WWSHtestcndc0.to_csv("MLPRWWSHAllcndc0tru.csv")
WWSHtestcndc1 = pd.DataFrame(WWSHtestcndc1)
WWSHtestcndc1.to_csv("MLPRWWSHAllcndc1tru.csv")
WWSHtestcndc2 = pd.DataFrame(WWSHtestcndc2)
WWSHtestcndc2.to_csv("MLPRWWSHAllcndc2tru.csv")
WWSHtestcndc3 = pd.DataFrame(WWSHtestcndc3)
WWSHtestcndc3.to_csv("MLPRWWSHAllcndc3tru.csv")
WWSHtestcndc4 = pd.DataFrame(WWSHtestcndc4)
WWSHtestcndc4.to_csv("MLPRWWSHAllcndc4tru.csv")




WWSHxtemptrain = pd.concat([WWSHTraindata2,WWSHTraindataxdartemp2],axis=1)
WWSHxtemptest = pd.concat([WWSHTestdata2,WWSHtestdataxdartemp2],axis=1)
for i in range(0,5):
    WWSHdata = WWSHxtemptrain.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    temptrain = WWSHdata.iloc[:,[1,2,3,4,5,6,7]]
    
    regressorWWSHxtemp = MLPRegressor()
    regressorWWSHxtemp.fit(temptrain,WWSHtrain)
    
    WWSHdatatest = WWSHxtemptest.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHdatatest = WWSHdatatest.reset_index().iloc[:,1:27]
    globals()['WWSHtesttemp'+str(i)]  = WWSHdatatest.iloc[:,0]
    temptest = WWSHdatatest.iloc[:,[1,2,3,4,5,6,7]]
    globals()['predtemp'+str(i)] = regressorWWSHxtemp.predict(temptest)


predtemp0 = pd.DataFrame(predtemp0)
predtemp0.to_csv("MLPRWWSHAlltemp0.csv")
predtemp1 = pd.DataFrame(predtemp1)
predtemp1.to_csv("MLPRWWSHAlltemp1.csv")
predtemp2 = pd.DataFrame(predtemp2)
predtemp2.to_csv("MLPRWWSHAlltemp2.csv")
predtemp3 = pd.DataFrame(predtemp3)
predtemp3.to_csv("MLPRWWSHAlltemp3.csv")
predtemp4 = pd.DataFrame(predtemp4)
predtemp4.to_csv("MLPRWWSHAlltemp4.csv")

WWSHtesttemp0 = pd.DataFrame(WWSHtesttemp0)
WWSHtesttemp0.to_csv("MLPRWWSHAlltemp0tru.csv")
WWSHtesttemp1 = pd.DataFrame(WWSHtesttemp1)
WWSHtesttemp1.to_csv("MLPRWWSHAlltemp1tru.csv")
WWSHtesttemp2 = pd.DataFrame(WWSHtesttemp2)
WWSHtesttemp2.to_csv("MLPRWWSHAlltemp2tru.csv")
WWSHtesttemp3 = pd.DataFrame(WWSHtesttemp3)
WWSHtesttemp3.to_csv("MLPRWWSHAlltemp3tru.csv")
WWSHtesttemp4 = pd.DataFrame(WWSHtesttemp4)
WWSHtesttemp4.to_csv("MLPRWWSHAlltemp4tru.csv")




WWSHxpartrain = pd.concat([WWSHTraindata2,WWSHTraindataxpar2],axis=1)
WWSHxpartest = pd.concat([WWSHTestdata2,WWSHtestdataxpar2],axis=1)
for i in range(0,5):
    WWSHdata = WWSHxpartrain.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    partrain = WWSHdata.iloc[:,[1,2,3,4,5,6,7]]
    
    regressorWWSHxpar = MLPRegressor()
    regressorWWSHxpar.fit(partrain,WWSHtrain)
    
    WWSHdatatest = WWSHxpartest.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHdatatest = WWSHdatatest.reset_index().iloc[:,1:27]
    globals()['WWSHtestpar'+str(i)]  = WWSHdatatest.iloc[:,0]
    partest = WWSHdatatest.iloc[:,[1,2,3,4,5,6,7]]
    globals()['predpar'+str(i)] = regressorWWSHxpar.predict(partest)


predpar0 = pd.DataFrame(predpar0)
predpar0.to_csv("MLPRWWSHAllpar0.csv")
predpar1 = pd.DataFrame(predpar1)
predpar1.to_csv("MLPRWWSHAllpar1.csv")
predpar2 = pd.DataFrame(predpar2)
predpar2.to_csv("MLPRWWSHAllpar2.csv")
predpar3 = pd.DataFrame(predpar3)
predpar3.to_csv("MLPRWWSHAllpar3.csv")
predpar4 = pd.DataFrame(predpar4)
predpar4.to_csv("MLPRWWSHAllpar4.csv")

WWSHtestpar0 = pd.DataFrame(WWSHtestpar0)
WWSHtestpar0.to_csv("MLPRWWSHAllpar0tru.csv")
WWSHtestpar1 = pd.DataFrame(WWSHtestpar1)
WWSHtestpar1.to_csv("MLPRWWSHAllpar1tru.csv")
WWSHtestpar2 = pd.DataFrame(WWSHtestpar2)
WWSHtestpar2.to_csv("MLPRWWSHAllpar2tru.csv")
WWSHtestpar3 = pd.DataFrame(WWSHtestpar3)
WWSHtestpar3.to_csv("MLPRWWSHAllpar3tru.csv")
WWSHtestpar4 = pd.DataFrame(WWSHtestpar4)
WWSHtestpar4.to_csv("MLPRWWSHAllpar4tru.csv")



WWSHxturbtrain = pd.concat([WWSHTraindata2,WWSHTraindataxturb2],axis=1)
WWSHxturbtest = pd.concat([WWSHTestdata2,WWSHtestdataxturb2],axis=1)
for i in range(0,5):
    WWSHdata = WWSHxturbtrain.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    turbtrain = WWSHdata.iloc[:,[1,2,3,4,5,6,7]]
    
    regressorWWSHxturb = MLPRegressor()
    regressorWWSHxturb.fit(turbtrain,WWSHtrain)
    
    WWSHdatatest = WWSHxturbtest.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHdatatest = WWSHdatatest.reset_index().iloc[:,1:27]
    globals()['WWSHtestturb'+str(i)]  = WWSHdatatest.iloc[:,0]
    turbtest = WWSHdatatest.iloc[:,[1,2,3,4,5,6,7]]
    globals()['predturb'+str(i)] = regressorWWSHxturb.predict(turbtest)


predturb0 = pd.DataFrame(predturb0)
predturb0.to_csv("MLPRWWSHAllturb0.csv")
predturb1 = pd.DataFrame(predturb1)
predturb1.to_csv("MLPRWWSHAllturb1.csv")
predturb2 = pd.DataFrame(predturb2)
predturb2.to_csv("MLPRWWSHAllturb2.csv")
predturb3 = pd.DataFrame(predturb3)
predturb3.to_csv("MLPRWWSHAllturb3.csv")
predturb4 = pd.DataFrame(predturb4)
predturb4.to_csv("MLPRWWSHAllturb4.csv")

WWSHtestturb0 = pd.DataFrame(WWSHtestturb0)
WWSHtestturb0.to_csv("MLPRWWSHAllturb0tru.csv")
WWSHtestturb1 = pd.DataFrame(WWSHtestturb1)
WWSHtestturb1.to_csv("MLPRWWSHAllturb1tru.csv")
WWSHtestturb2 = pd.DataFrame(WWSHtestturb2)
WWSHtestturb2.to_csv("MLPRWWSHAllturb2tru.csv")
WWSHtestturb3 = pd.DataFrame(WWSHtestturb3)
WWSHtestturb3.to_csv("MLPRWWSHAllturb3tru.csv")
WWSHtestturb4 = pd.DataFrame(WWSHtestturb4)
WWSHtestturb4.to_csv("MLPRWWSHAllturb4tru.csv")


WWSHxcphltrain = pd.concat([WWSHTraindata2,WWSHTraindataxcphl2],axis=1)
WWSHxcphltest = pd.concat([WWSHTestdata2,WWSHtestdataxcphl2],axis=1)
for i in range(0,5):
    WWSHdata = WWSHxcphltrain.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdata = WWSHdata.dropna()
    WWSHtrain = WWSHdata.iloc[:,0]
    cphltrain = WWSHdata.iloc[:,[1,2,3,4,5,6,7]]
    
    regressorWWSHxcphl = MLPRegressor()
    regressorWWSHxcphl.fit(cphltrain,WWSHtrain)
    
    WWSHdatatest = WWSHxcphltest.iloc[:,[i,i*7+5,i*7+6,i*7+7,i*7+8,i*7+9,i*7+10,i*7+11]]
    WWSHdatatest = WWSHdatatest.dropna()
    WWSHdatatest = WWSHdatatest.reset_index().iloc[:,1:27]
    globals()['WWSHtestcphl'+str(i)]  = WWSHdatatest.iloc[:,0]
    cphltest = WWSHdatatest.iloc[:,[1,2,3,4,5,6,7]]
    globals()['predcphl'+str(i)] = regressorWWSHxcphl.predict(cphltest)


predcphl0 = pd.DataFrame(predcphl0)
predcphl0.to_csv("MLPRWWSHAllcphl0.csv")
predcphl1 = pd.DataFrame(predcphl1)
predcphl1.to_csv("MLPRWWSHAllcphl1.csv")
predcphl2 = pd.DataFrame(predcphl2)
predcphl2.to_csv("MLPRWWSHAllcphl2.csv")
predcphl3 = pd.DataFrame(predcphl3)
predcphl3.to_csv("MLPRWWSHAllcphl3.csv")
predcphl4 = pd.DataFrame(predcphl4)
predcphl4.to_csv("MLPRWWSHAllcphl4.csv")

WWSHtestcphl0 = pd.DataFrame(WWSHtestcphl0)
WWSHtestcphl0.to_csv("MLPRWWSHAllcphl0tru.csv")
WWSHtestcphl1 = pd.DataFrame(WWSHtestcphl1)
WWSHtestcphl1.to_csv("MLPRWWSHAllcphl1tru.csv")
WWSHtestcphl2 = pd.DataFrame(WWSHtestcphl2)
WWSHtestcphl2.to_csv("MLPRWWSHAllcphl2tru.csv")
WWSHtestcphl3 = pd.DataFrame(WWSHtestcphl3)
WWSHtestcphl3.to_csv("MLPRWWSHAllcphl3tru.csv")
WWSHtestcphl4 = pd.DataFrame(WWSHtestcphl4)
WWSHtestcphl4.to_csv("MLPRWWSHAllcphl4tru.csv")

