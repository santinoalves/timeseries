# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:29:47 2020

@author: Connor Simpson
"""



import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


WINDTestdata = pd.read_csv("WINDTestdata.csv")
WINDTraindata = pd.read_csv("WINDTraindata.csv")

WINDTraindataxCNDC = pd.read_csv("WINDTraindataxCNDC.csv")
WINDTraindataxPAR = pd.read_csv("WINDTraindataxPAR.csv")
WINDTraindataxTURB = pd.read_csv("WINDTraindataxTURB.csv")
WINDTraindataxCPHL = pd.read_csv("WINDTraindataxCPHL.csv")
WINDTraindataxTEMP = pd.read_csv("WINDTraindataxTEMP.csv")


WINDtestdataxTURB = pd.read_csv("WINDtestdataxTURB.csv")
WINDtestdataxPAR = pd.read_csv("WINDtestdataxPAR.csv")
WINDtestdataxCNDC = pd.read_csv("WINDtestdataxCNDC.csv")
WINDtestdataxCPHL = pd.read_csv("WINDtestdataxCPHL.csv")
WINDtestdataxTEMP = pd.read_csv("WINDtestdataxTEMP.csv")
#################################################

WINDTestdata = WINDTestdata.iloc[:,1:]
WINDTraindata = WINDTraindata.iloc[:,1:]

WINDTraindataxCNDC = WINDTraindataxCNDC.iloc[:,1:]
WINDTraindataxPAR = WINDTraindataxPAR.iloc[:,1:]
WINDTraindataxTURB = WINDTraindataxTURB.iloc[:,1:]
WINDTraindataxCPHL = WINDTraindataxCPHL.iloc[:,1:]
WINDTraindataxTEMP = WINDTraindataxTEMP.iloc[:,1:]

WINDtestdataxTURB = WINDtestdataxTURB.iloc[:,1:]
WINDtestdataxPAR = WINDtestdataxPAR.iloc[:,1:]
WINDtestdataxCNDC = WINDtestdataxCNDC.iloc[:,1:]
WINDtestdataxCPHL = WINDtestdataxCPHL.iloc[:,1:]
WINDtestdataxTEMP = WINDtestdataxTEMP.iloc[:,1:]

#################################################




WINDxWINDtrain = pd.concat([WINDTraindata,WINDTraindataxCNDC,WINDTraindataxPAR,WINDTraindataxTURB,WINDTraindataxCPHL,WINDTraindataxTEMP],axis=1)
WINDxWINDtest = pd.concat([WINDTestdata,WINDtestdataxCNDC,WINDtestdataxPAR,WINDtestdataxTURB,WINDtestdataxCPHL,WINDtestdataxTEMP],axis=1)
for i in range(0,5):
    WINDdata = WINDxWINDtrain.iloc[:,[i,i*2+5,i*2+6,i*2+15,i*2+16,i*2+25,i*2+26,i*2+35,i*2+36,i*2+45,i*2+46]]
    WINDdata = WINDdata.dropna()
    WINDtrain = WINDdata.iloc[:,0]
    WINDtrainx = WINDdata.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
    
    regressorWINDxWIND = MLPRegressor()
    regressorWINDxWIND.fit(WINDtrainx,WINDtrain)
    
    WINDdatatest = WINDxWINDtest.iloc[:,[i,i*2+5,i*2+6,i*2+15,i*2+16,i*2+25,i*2+26,i*2+35,i*2+36,i*2+45,i*2+46]]
    WINDdatatest = WINDdatatest.dropna()
    WINDdatatest = WINDdatatest.reset_index().iloc[:,1:]
    globals()['WINDtest'+str(i)]  = WINDdatatest.iloc[:,0]
    WINDtest = WINDdatatest.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
    globals()['pred'+str(i)] = regressorWINDxWIND.predict(WINDtest)


pred0 = pd.DataFrame(pred0)
pred0.to_csv("WIND0.csv")
pred1 = pd.DataFrame(pred1)
pred1.to_csv("WIND1.csv")
pred2 = pd.DataFrame(pred2)
pred2.to_csv("WIND2.csv")
pred3 = pd.DataFrame(pred3)
pred3.to_csv("WIND3.csv")
pred4 = pd.DataFrame(pred4)
pred4.to_csv("WIND4.csv")

WINDtest0 = pd.DataFrame(WINDtest0)
WINDtest0.to_csv("WIND0tru.csv")
WINDtest1 = pd.DataFrame(WINDtest1)
WINDtest1.to_csv("WIND1tru.csv")
WINDtest2 = pd.DataFrame(WINDtest2)
WINDtest2.to_csv("WIND2tru.csv")
WINDtest3 = pd.DataFrame(WINDtest3)
WINDtest3.to_csv("WIND3tru.csv")
WINDtest4 = pd.DataFrame(WINDtest4)
WINDtest4.to_csv("WIND4tru.csv")


WINDxWINDtrain = pd.concat([WINDTraindata,WINDTraindataxCNDC],axis=1)
WINDxWINDtest = pd.concat([WINDTestdata,WINDtestdataxCNDC],axis=1)
for i in range(0,5):
    WINDdata = WINDxWINDtrain.iloc[:,[i,i*2+5,i*2+6]]
    WINDdata = WINDdata.dropna()
    WINDtrain = WINDdata.iloc[:,0]
    WINDtrainx = WINDdata.iloc[:,[1,2]]
    
    regressorWINDxWIND = MLPRegressor()
    regressorWINDxWIND.fit(WINDtrainx,WINDtrain)
    
    WINDdatatest = WINDxWINDtest.iloc[:,[i,i*2+5,i*2+6]]
    WINDdatatest = WINDdatatest.dropna()
    WINDdatatest = WINDdatatest.reset_index().iloc[:,1:]
    globals()['WINDtestCNDC'+str(i)]  = WINDdatatest.iloc[:,0]
    WINDtest = WINDdatatest.iloc[:,[1,2]]
    globals()['predCNDC'+str(i)] = regressorWINDxWIND.predict(WINDtest)


pred0 = pd.DataFrame(predCNDC0)
pred0.to_csv("WINDCNDC0.csv")
pred1 = pd.DataFrame(predCNDC1)
pred1.to_csv("WINDCNDC1.csv")
pred2 = pd.DataFrame(predCNDC2)
pred2.to_csv("WINDCNDC2.csv")
pred3 = pd.DataFrame(predCNDC3)
pred3.to_csv("WINDCNDC3.csv")
pred4 = pd.DataFrame(predCNDC4)
pred4.to_csv("WINDCNDC4.csv")

WINDtest0 = pd.DataFrame(WINDtestCNDC0)
WINDtest0.to_csv("WINDCNDC0tru.csv")
WINDtest1 = pd.DataFrame(WINDtestCNDC1)
WINDtest1.to_csv("WINDCNDC1tru.csv")
WINDtest2 = pd.DataFrame(WINDtestCNDC2)
WINDtest2.to_csv("WINDCNDC2tru.csv")
WINDtest3 = pd.DataFrame(WINDtestCNDC3)
WINDtest3.to_csv("WINDCNDC3tru.csv")
WINDtest4 = pd.DataFrame(WINDtestCNDC4)
WINDtest4.to_csv("WINDCNDC4tru.csv")




WINDxWINDtrain = pd.concat([WINDTraindata,WINDTraindataxTEMP],axis=1)
WINDxWINDtest = pd.concat([WINDTestdata,WINDtestdataxTEMP],axis=1)
for i in range(0,5):
    WINDdata = WINDxWINDtrain.iloc[:,[i,i*2+5,i*2+6]]
    WINDdata = WINDdata.dropna()
    WINDtrain = WINDdata.iloc[:,0]
    WINDtrainx = WINDdata.iloc[:,[1,2]]
    
    regressorWINDxWIND = MLPRegressor()
    regressorWINDxWIND.fit(WINDtrainx,WINDtrain)
    
    WINDdatatest = WINDxWINDtest.iloc[:,[i,i*2+5,i*2+6]]
    WINDdatatest = WINDdatatest.dropna()
    WINDdatatest = WINDdatatest.reset_index().iloc[:,1:]
    globals()['WINDtestTEMP'+str(i)]  = WINDdatatest.iloc[:,0]
    WINDtest = WINDdatatest.iloc[:,[1,2]]
    globals()['predTEMP'+str(i)] = regressorWINDxWIND.predict(WINDtest)


pred0 = pd.DataFrame(predTEMP0)
pred0.to_csv("WINDTEMP0.csv")
pred1 = pd.DataFrame(predTEMP1)
pred1.to_csv("WINDTEMP1.csv")
pred2 = pd.DataFrame(predTEMP2)
pred2.to_csv("WINDTEMP2.csv")
pred3 = pd.DataFrame(predTEMP3)
pred3.to_csv("WINDTEMP3.csv")
pred4 = pd.DataFrame(predTEMP4)
pred4.to_csv("WINDTEMP4.csv")

WINDtest0 = pd.DataFrame(WINDtestTEMP0)
WINDtest0.to_csv("WINDTEMP0tru.csv")
WINDtest1 = pd.DataFrame(WINDtestTEMP1)
WINDtest1.to_csv("WINDTEMP1tru.csv")
WINDtest2 = pd.DataFrame(WINDtestTEMP2)
WINDtest2.to_csv("WINDTEMP2tru.csv")
WINDtest3 = pd.DataFrame(WINDtestTEMP3)
WINDtest3.to_csv("WINDTEMP3tru.csv")
WINDtest4 = pd.DataFrame(WINDtestTEMP4)
WINDtest4.to_csv("WINDTEMP4tru.csv")




WINDxWINDtrain = pd.concat([WINDTraindata,WINDTraindataxTURB],axis=1)
WINDxWINDtest = pd.concat([WINDTestdata,WINDtestdataxTURB],axis=1)
for i in range(0,5):
    WINDdata = WINDxWINDtrain.iloc[:,[i,i*2+5,i*2+6]]
    WINDdata = WINDdata.dropna()
    WINDtrain = WINDdata.iloc[:,0]
    WINDtrainx = WINDdata.iloc[:,[1,2]]
    
    regressorWINDxWIND = MLPRegressor()
    regressorWINDxWIND.fit(WINDtrainx,WINDtrain)
    
    WINDdatatest = WINDxWINDtest.iloc[:,[i,i*2+5,i*2+6]]
    WINDdatatest = WINDdatatest.dropna()
    WINDdatatest = WINDdatatest.reset_index().iloc[:,1:]
    globals()['WINDtestTURB'+str(i)]  = WINDdatatest.iloc[:,0]
    WINDtest = WINDdatatest.iloc[:,[1,2]]
    globals()['predTURB'+str(i)] = regressorWINDxWIND.predict(WINDtest)


pred0 = pd.DataFrame(predTURB0)
pred0.to_csv("WINDTURB0.csv")
pred1 = pd.DataFrame(predTURB1)
pred1.to_csv("WINDTURB1.csv")
pred2 = pd.DataFrame(predTURB2)
pred2.to_csv("WINDTURB2.csv")
pred3 = pd.DataFrame(predTURB3)
pred3.to_csv("WINDTURB3.csv")
pred4 = pd.DataFrame(predTURB4)
pred4.to_csv("WINDTURB4.csv")

WINDtest0 = pd.DataFrame(WINDtestTURB0)
WINDtest0.to_csv("WINDTURB0tru.csv")
WINDtest1 = pd.DataFrame(WINDtestTURB1)
WINDtest1.to_csv("WINDTURB1tru.csv")
WINDtest2 = pd.DataFrame(WINDtestTURB2)
WINDtest2.to_csv("WINDTURB2tru.csv")
WINDtest3 = pd.DataFrame(WINDtestTURB3)
WINDtest3.to_csv("WINDTURB3tru.csv")
WINDtest4 = pd.DataFrame(WINDtestTURB4)
WINDtest4.to_csv("WINDTURB4tru.csv")



WINDxWINDtrain = pd.concat([WINDTraindata,WINDTraindataxPAR],axis=1)
WINDxWINDtest = pd.concat([WINDTestdata,WINDtestdataxPAR],axis=1)
for i in range(0,5):
    WINDdata = WINDxWINDtrain.iloc[:,[i,i*2+5,i*2+6]]
    WINDdata = WINDdata.dropna()
    WINDtrain = WINDdata.iloc[:,0]
    WINDtrainx = WINDdata.iloc[:,[1,2]]
    
    regressorWINDxWIND = MLPRegressor()
    regressorWINDxWIND.fit(WINDtrainx,WINDtrain)
    
    WINDdatatest = WINDxWINDtest.iloc[:,[i,i*2+5,i*2+6]]
    WINDdatatest = WINDdatatest.dropna()
    WINDdatatest = WINDdatatest.reset_index().iloc[:,1:]
    globals()['WINDtestPAR'+str(i)]  = WINDdatatest.iloc[:,0]
    WINDtest = WINDdatatest.iloc[:,[1,2]]
    globals()['predPAR'+str(i)] = regressorWINDxWIND.predict(WINDtest)


pred0 = pd.DataFrame(predPAR0)
pred0.to_csv("WINDPAR0.csv")
pred1 = pd.DataFrame(predPAR1)
pred1.to_csv("WINDPAR1.csv")
pred2 = pd.DataFrame(predPAR2)
pred2.to_csv("WINDPAR2.csv")
pred3 = pd.DataFrame(predPAR3)
pred3.to_csv("WINDPAR3.csv")
pred4 = pd.DataFrame(predPAR4)
pred4.to_csv("WINDPAR4.csv")

WINDtest0 = pd.DataFrame(WINDtestPAR0)
WINDtest0.to_csv("WINDPAR0tru.csv")
WINDtest1 = pd.DataFrame(WINDtestPAR1)
WINDtest1.to_csv("WINDPAR1tru.csv")
WINDtest2 = pd.DataFrame(WINDtestPAR2)
WINDtest2.to_csv("WINDPAR2tru.csv")
WINDtest3 = pd.DataFrame(WINDtestPAR3)
WINDtest3.to_csv("WINDPAR3tru.csv")
WINDtest4 = pd.DataFrame(WINDtestPAR4)
WINDtest4.to_csv("WINDPAR4tru.csv")



WINDxWINDtrain = pd.concat([WINDTraindata,WINDTraindataxCPHL],axis=1)
WINDxWINDtest = pd.concat([WINDTestdata,WINDtestdataxCPHL],axis=1)
for i in range(0,5):
    WINDdata = WINDxWINDtrain.iloc[:,[i,i*2+5,i*2+6]]
    WINDdata = WINDdata.dropna()
    WINDtrain = WINDdata.iloc[:,0]
    WINDtrainx = WINDdata.iloc[:,[1,2]]
    
    regressorWINDxWIND = MLPRegressor()
    regressorWINDxWIND.fit(WINDtrainx,WINDtrain)
    
    WINDdatatest = WINDxWINDtest.iloc[:,[i,i*2+5,i*2+6]]
    WINDdatatest = WINDdatatest.dropna()
    WINDdatatest = WINDdatatest.reset_index().iloc[:,1:]
    globals()['WINDtestCPHL'+str(i)]  = WINDdatatest.iloc[:,0]
    WINDtest = WINDdatatest.iloc[:,[1,2]]
    globals()['predCPHL'+str(i)] = regressorWINDxWIND.predict(WINDtest)


pred0 = pd.DataFrame(predCPHL0)
pred0.to_csv("WINDCPHL0.csv")
pred1 = pd.DataFrame(predCPHL1)
pred1.to_csv("WINDCPHL1.csv")
pred2 = pd.DataFrame(predCPHL2)
pred2.to_csv("WINDCPHL2.csv")
pred3 = pd.DataFrame(predCPHL3)
pred3.to_csv("WINDCPHL3.csv")
pred4 = pd.DataFrame(predCPHL4)
pred4.to_csv("WINDCPHL4.csv")

WINDtest0 = pd.DataFrame(WINDtestCPHL0)
WINDtest0.to_csv("WINDCPHL0tru.csv")
WINDtest1 = pd.DataFrame(WINDtestCPHL1)
WINDtest1.to_csv("WINDCPHL1tru.csv")
WINDtest2 = pd.DataFrame(WINDtestCPHL2)
WINDtest2.to_csv("WINDCPHL2tru.csv")
WINDtest3 = pd.DataFrame(WINDtestCPHL3)
WINDtest3.to_csv("WINDCPHL3tru.csv")
WINDtest4 = pd.DataFrame(WINDtestCPHL4)
WINDtest4.to_csv("WINDCPHL4tru.csv")


