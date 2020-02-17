# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:38:12 2020

@author: Connor Simpson
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


CNDCTestdata = pd.read_csv("CNDCTestdata.csv")
CNDCTraindata = pd.read_csv("CNDCTraindata.csv")

CNDCTraindataxPAR = pd.read_csv("CNDCTraindataxPAR.csv")
CNDCTraindataxTURB = pd.read_csv("CNDCTraindataxTURB.csv")
CNDCTraindataxCPHL = pd.read_csv("CNDCTraindataxCPHL.csv")
CNDCTraindataxTEMP = pd.read_csv("CNDCTraindataxTEMP.csv")


CNDCtestdataxTURB = pd.read_csv("CNDCtestdataxTURB.csv")
CNDCtestdataxPAR = pd.read_csv("CNDCtestdataxPAR.csv")
CNDCtestdataxCPHL = pd.read_csv("CNDCtestdataxCPHL.csv")
CNDCtestdataxTEMP = pd.read_csv("CNDCtestdataxTEMP.csv")
#################################################

CNDCTestdata = CNDCTestdata.iloc[:,1:]
CNDCTraindata = CNDCTraindata.iloc[:,1:]

CNDCTraindataxPAR = CNDCTraindataxPAR.iloc[:,1:]
CNDCTraindataxTURB = CNDCTraindataxTURB.iloc[:,1:]
CNDCTraindataxCPHL = CNDCTraindataxCPHL.iloc[:,1:]
CNDCTraindataxTEMP = CNDCTraindataxTEMP.iloc[:,1:]

CNDCtestdataxTURB = CNDCtestdataxTURB.iloc[:,1:]
CNDCtestdataxPAR = CNDCtestdataxPAR.iloc[:,1:]
CNDCtestdataxCPHL = CNDCtestdataxCPHL.iloc[:,1:]
CNDCtestdataxTEMP = CNDCtestdataxTEMP.iloc[:,1:]

#################################################




CNDCxCNDCtrain = pd.concat([CNDCTraindata,CNDCTraindataxPAR,CNDCTraindataxTURB,CNDCTraindataxCPHL,CNDCTraindataxTEMP],axis=1)
CNDCxCNDCtest = pd.concat([CNDCTestdata,CNDCtestdataxPAR,CNDCtestdataxTURB,CNDCtestdataxCPHL,CNDCtestdataxTEMP],axis=1)
for i in range(0,5):
    CNDCdata = CNDCxCNDCtrain.iloc[:,[i,i+5,i+10,i+15,i+20]]
    CNDCdata = CNDCdata.dropna()
    CNDCtrain = CNDCdata.iloc[:,0]
    CNDCtrainx = CNDCdata.iloc[:,[1,2,3,4]]
    
    regressorCNDCxCNDC = MLPRegressor()
    regressorCNDCxCNDC.fit(CNDCtrainx,CNDCtrain)
    
    CNDCdatatest = CNDCxCNDCtest.iloc[:,[i,i+5,i+10,i+15,i+20]]
    CNDCdatatest = CNDCdatatest.dropna()
    CNDCdatatest = CNDCdatatest.reset_index().iloc[:,1:]
    globals()['CNDCtest'+str(i)]  = CNDCdatatest.iloc[:,0]
    CNDCtest = CNDCdatatest.iloc[:,[1,2,3,4]]
    globals()['pred'+str(i)] = regressorCNDCxCNDC.predict(CNDCtest)


pred0 = pd.DataFrame(pred0)
pred0.to_csv("CNDC0.csv")
pred1 = pd.DataFrame(pred1)
pred1.to_csv("CNDC1.csv")
pred2 = pd.DataFrame(pred2)
pred2.to_csv("CNDC2.csv")
pred3 = pd.DataFrame(pred3)
pred3.to_csv("CNDC3.csv")
pred4 = pd.DataFrame(pred4)
pred4.to_csv("CNDC4.csv")

CNDCtest0 = pd.DataFrame(CNDCtest0)
CNDCtest0.to_csv("CNDC0tru.csv")
CNDCtest1 = pd.DataFrame(CNDCtest1)
CNDCtest1.to_csv("CNDC1tru.csv")
CNDCtest2 = pd.DataFrame(CNDCtest2)
CNDCtest2.to_csv("CNDC2tru.csv")
CNDCtest3 = pd.DataFrame(CNDCtest3)
CNDCtest3.to_csv("CNDC3tru.csv")
CNDCtest4 = pd.DataFrame(CNDCtest4)
CNDCtest4.to_csv("CNDC4tru.csv")


CNDCxCNDCtrain = pd.concat([CNDCTraindata,CNDCTraindataxTEMP],axis=1)
CNDCxCNDCtest = pd.concat([CNDCTestdata,CNDCtestdataxTEMP],axis=1)
for i in range(0,5):
    CNDCdata = CNDCxCNDCtrain.iloc[:,[i,i+5]]
    CNDCdata = CNDCdata.dropna()
    CNDCtrain = CNDCdata.iloc[:,0]
    CNDCtrainx = CNDCdata.iloc[:,[1]]
    
    regressorCNDCxCNDC = MLPRegressor()
    regressorCNDCxCNDC.fit(CNDCtrainx,CNDCtrain)
    
    CNDCdatatest = CNDCxCNDCtest.iloc[:,[i,i+5]]
    CNDCdatatest = CNDCdatatest.dropna()
    CNDCdatatest = CNDCdatatest.reset_index().iloc[:,1:]
    globals()['CNDCtestTEMP'+str(i)]  = CNDCdatatest.iloc[:,0]
    CNDCtest = CNDCdatatest.iloc[:,[1]]
    globals()['predTEMP'+str(i)] = regressorCNDCxCNDC.predict(CNDCtest)


pred0 = pd.DataFrame(predTEMP0)
pred0.to_csv("CNDCTEMP0.csv")
pred1 = pd.DataFrame(predTEMP1)
pred1.to_csv("CNDCTEMP1.csv")
pred2 = pd.DataFrame(predTEMP2)
pred2.to_csv("CNDCTEMP2.csv")
pred3 = pd.DataFrame(predTEMP3)
pred3.to_csv("CNDCTEMP3.csv")
pred4 = pd.DataFrame(predTEMP4)
pred4.to_csv("CNDCTEMP4.csv")

CNDCtest0 = pd.DataFrame(CNDCtestTEMP0)
CNDCtest0.to_csv("CNDCTEMP0tru.csv")
CNDCtest1 = pd.DataFrame(CNDCtestTEMP1)
CNDCtest1.to_csv("CNDCTEMP1tru.csv")
CNDCtest2 = pd.DataFrame(CNDCtestTEMP2)
CNDCtest2.to_csv("CNDCTEMP2tru.csv")
CNDCtest3 = pd.DataFrame(CNDCtestTEMP3)
CNDCtest3.to_csv("CNDCTEMP3tru.csv")
CNDCtest4 = pd.DataFrame(CNDCtestTEMP4)
CNDCtest4.to_csv("CNDCTEMP4tru.csv")



CNDCxCNDCtrain = pd.concat([CNDCTraindata,CNDCTraindataxTURB],axis=1)
CNDCxCNDCtest = pd.concat([CNDCTestdata,CNDCtestdataxTURB],axis=1)
for i in range(0,5):
    CNDCdata = CNDCxCNDCtrain.iloc[:,[i,i+5]]
    CNDCdata = CNDCdata.dropna()
    CNDCtrain = CNDCdata.iloc[:,0]
    CNDCtrainx = CNDCdata.iloc[:,[1]]
    
    regressorCNDCxCNDC = MLPRegressor()
    regressorCNDCxCNDC.fit(CNDCtrainx,CNDCtrain)
    
    CNDCdatatest = CNDCxCNDCtest.iloc[:,[i,i+5]]
    CNDCdatatest = CNDCdatatest.dropna()
    CNDCdatatest = CNDCdatatest.reset_index().iloc[:,1:]
    globals()['CNDCtestTURB'+str(i)]  = CNDCdatatest.iloc[:,0]
    CNDCtest = CNDCdatatest.iloc[:,[1]]
    globals()['predTURB'+str(i)] = regressorCNDCxCNDC.predict(CNDCtest)


pred0 = pd.DataFrame(predTURB0)
pred0.to_csv("CNDCTURB0.csv")
pred1 = pd.DataFrame(predTURB1)
pred1.to_csv("CNDCTURB1.csv")
pred2 = pd.DataFrame(predTURB2)
pred2.to_csv("CNDCTURB2.csv")
pred3 = pd.DataFrame(predTURB3)
pred3.to_csv("CNDCTURB3.csv")
pred4 = pd.DataFrame(predTURB4)
pred4.to_csv("CNDCTURB4.csv")

CNDCtest0 = pd.DataFrame(CNDCtestTURB0)
CNDCtest0.to_csv("CNDCTURB0tru.csv")
CNDCtest1 = pd.DataFrame(CNDCtestTURB1)
CNDCtest1.to_csv("CNDCTURB1tru.csv")
CNDCtest2 = pd.DataFrame(CNDCtestTURB2)
CNDCtest2.to_csv("CNDCTURB2tru.csv")
CNDCtest3 = pd.DataFrame(CNDCtestTURB3)
CNDCtest3.to_csv("CNDCTURB3tru.csv")
CNDCtest4 = pd.DataFrame(CNDCtestTURB4)
CNDCtest4.to_csv("CNDCTURB4tru.csv")



CNDCxCNDCtrain = pd.concat([CNDCTraindata,CNDCTraindataxPAR],axis=1)
CNDCxCNDCtest = pd.concat([CNDCTestdata,CNDCtestdataxPAR],axis=1)
for i in range(0,5):
    CNDCdata = CNDCxCNDCtrain.iloc[:,[i,i+5]]
    CNDCdata = CNDCdata.dropna()
    CNDCtrain = CNDCdata.iloc[:,0]
    CNDCtrainx = CNDCdata.iloc[:,[1]]
    
    regressorCNDCxCNDC = MLPRegressor()
    regressorCNDCxCNDC.fit(CNDCtrainx,CNDCtrain)
    
    CNDCdatatest = CNDCxCNDCtest.iloc[:,[i,i+5]]
    CNDCdatatest = CNDCdatatest.dropna()
    CNDCdatatest = CNDCdatatest.reset_index().iloc[:,1:]
    globals()['CNDCtestPAR'+str(i)]  = CNDCdatatest.iloc[:,0]
    CNDCtest = CNDCdatatest.iloc[:,[1]]
    globals()['predPAR'+str(i)] = regressorCNDCxCNDC.predict(CNDCtest)


pred0 = pd.DataFrame(predPAR0)
pred0.to_csv("CNDCPAR0.csv")
pred1 = pd.DataFrame(predPAR1)
pred1.to_csv("CNDCPAR1.csv")
pred2 = pd.DataFrame(predPAR2)
pred2.to_csv("CNDCPAR2.csv")
pred3 = pd.DataFrame(predPAR3)
pred3.to_csv("CNDCPAR3.csv")
pred4 = pd.DataFrame(predPAR4)
pred4.to_csv("CNDCPAR4.csv")

CNDCtest0 = pd.DataFrame(CNDCtestPAR0)
CNDCtest0.to_csv("CNDCPAR0tru.csv")
CNDCtest1 = pd.DataFrame(CNDCtestPAR1)
CNDCtest1.to_csv("CNDCPAR1tru.csv")
CNDCtest2 = pd.DataFrame(CNDCtestPAR2)
CNDCtest2.to_csv("CNDCPAR2tru.csv")
CNDCtest3 = pd.DataFrame(CNDCtestPAR3)
CNDCtest3.to_csv("CNDCPAR3tru.csv")
CNDCtest4 = pd.DataFrame(CNDCtestPAR4)
CNDCtest4.to_csv("CNDCPAR4tru.csv")



CNDCxCNDCtrain = pd.concat([CNDCTraindata,CNDCTraindataxCPHL],axis=1)
CNDCxCNDCtest = pd.concat([CNDCTestdata,CNDCtestdataxCPHL],axis=1)
for i in range(0,5):
    CNDCdata = CNDCxCNDCtrain.iloc[:,[i,i+5]]
    CNDCdata = CNDCdata.dropna()
    CNDCtrain = CNDCdata.iloc[:,0]
    CNDCtrainx = CNDCdata.iloc[:,[1]]
    
    regressorCNDCxCNDC = MLPRegressor()
    regressorCNDCxCNDC.fit(CNDCtrainx,CNDCtrain)
    
    CNDCdatatest = CNDCxCNDCtest.iloc[:,[i,i+5]]
    CNDCdatatest = CNDCdatatest.dropna()
    CNDCdatatest = CNDCdatatest.reset_index().iloc[:,1:]
    globals()['CNDCtestCPHL'+str(i)]  = CNDCdatatest.iloc[:,0]
    CNDCtest = CNDCdatatest.iloc[:,[1]]
    globals()['predCPHL'+str(i)] = regressorCNDCxCNDC.predict(CNDCtest)


pred0 = pd.DataFrame(predCPHL0)
pred0.to_csv("CNDCCPHL0.csv")
pred1 = pd.DataFrame(predCPHL1)
pred1.to_csv("CNDCCPHL1.csv")
pred2 = pd.DataFrame(predCPHL2)
pred2.to_csv("CNDCCPHL2.csv")
pred3 = pd.DataFrame(predCPHL3)
pred3.to_csv("CNDCCPHL3.csv")
pred4 = pd.DataFrame(predCPHL4)
pred4.to_csv("CNDCCPHL4.csv")

CNDCtest0 = pd.DataFrame(CNDCtestCPHL0)
CNDCtest0.to_csv("CNDCCPHL0tru.csv")
CNDCtest1 = pd.DataFrame(CNDCtestCPHL1)
CNDCtest1.to_csv("CNDCCPHL1tru.csv")
CNDCtest2 = pd.DataFrame(CNDCtestCPHL2)
CNDCtest2.to_csv("CNDCCPHL2tru.csv")
CNDCtest3 = pd.DataFrame(CNDCtestCPHL3)
CNDCtest3.to_csv("CNDCCPHL3tru.csv")
CNDCtest4 = pd.DataFrame(CNDCtestCPHL4)
CNDCtest4.to_csv("CNDCCPHL4tru.csv")



