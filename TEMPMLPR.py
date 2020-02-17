# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:19:12 2020

@author: Connor Simpson
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


TEMPTestdata = pd.read_csv("TEMPTestdata.csv")
TEMPTraindata = pd.read_csv("TEMPTraindata.csv")

TEMPTraindataxCNDC = pd.read_csv("TEMPTraindataxCNDC.csv")
TEMPTraindataxPAR = pd.read_csv("TEMPTraindataxPAR.csv")
TEMPTraindataxTURB = pd.read_csv("TEMPTraindataxTURB.csv")
TEMPTraindataxCPHL = pd.read_csv("TEMPTraindataxCPHL.csv")



TEMPtestdataxTURB = pd.read_csv("TEMPtestdataxTURB.csv")
TEMPtestdataxPAR = pd.read_csv("TEMPtestdataxPAR.csv")
TEMPtestdataxCNDC = pd.read_csv("TEMPtestdataxCNDC.csv")
TEMPtestdataxCPHL = pd.read_csv("TEMPtestdataxCPHL.csv")

#################################################

TEMPTestdata = TEMPTestdata.iloc[:,1:]
TEMPTraindata = TEMPTraindata.iloc[:,1:]

TEMPTraindataxCNDC = TEMPTraindataxCNDC.iloc[:,1:]
TEMPTraindataxPAR = TEMPTraindataxPAR.iloc[:,1:]
TEMPTraindataxTURB = TEMPTraindataxTURB.iloc[:,1:]
TEMPTraindataxCPHL = TEMPTraindataxCPHL.iloc[:,1:]


TEMPtestdataxTURB = TEMPtestdataxTURB.iloc[:,1:]
TEMPtestdataxPAR = TEMPtestdataxPAR.iloc[:,1:]
TEMPtestdataxCNDC = TEMPtestdataxCNDC.iloc[:,1:]
TEMPtestdataxCPHL = TEMPtestdataxCPHL.iloc[:,1:]


#################################################




TEMPxtemptrain = pd.concat([TEMPTraindata,TEMPTraindataxCNDC,TEMPTraindataxPAR,TEMPTraindataxTURB,TEMPTraindataxCPHL],axis=1)
TEMPxtemptest = pd.concat([TEMPTestdata,TEMPtestdataxCNDC,TEMPtestdataxPAR,TEMPtestdataxTURB,TEMPtestdataxCPHL],axis=1)
for i in range(0,5):
    TEMPdata = TEMPxtemptrain.iloc[:,[i,i+5,i+10,i+15,i+20]]
    TEMPdata = TEMPdata.dropna()
    TEMPtrain = TEMPdata.iloc[:,0]
    temptrain = TEMPdata.iloc[:,[1,2,3,4]]
    
    regressorTEMPxtemp = MLPRegressor()
    regressorTEMPxtemp.fit(temptrain,TEMPtrain)
    
    TEMPdatatest = TEMPxtemptest.iloc[:,[i,i+5,i+10,i+15,i+20]]
    TEMPdatatest = TEMPdatatest.dropna()
    TEMPdatatest = TEMPdatatest.reset_index().iloc[:,1:]
    globals()['TEMPtest'+str(i)]  = TEMPdatatest.iloc[:,0]
    temptest = TEMPdatatest.iloc[:,[1,2,3,4]]
    globals()['pred'+str(i)] = regressorTEMPxtemp.predict(temptest)


pred0 = pd.DataFrame(pred0)
pred0.to_csv("TEMP0.csv")
pred1 = pd.DataFrame(pred1)
pred1.to_csv("TEMP1.csv")
pred2 = pd.DataFrame(pred2)
pred2.to_csv("TEMP2.csv")
pred3 = pd.DataFrame(pred3)
pred3.to_csv("TEMP3.csv")
pred4 = pd.DataFrame(pred4)
pred4.to_csv("TEMP4.csv")

TEMPtest0 = pd.DataFrame(TEMPtest0)
TEMPtest0.to_csv("TEMP0tru.csv")
TEMPtest1 = pd.DataFrame(TEMPtest1)
TEMPtest1.to_csv("TEMP1tru.csv")
TEMPtest2 = pd.DataFrame(TEMPtest2)
TEMPtest2.to_csv("TEMP2tru.csv")
TEMPtest3 = pd.DataFrame(TEMPtest3)
TEMPtest3.to_csv("TEMP3tru.csv")
TEMPtest4 = pd.DataFrame(TEMPtest4)
TEMPtest4.to_csv("TEMP4tru.csv")



TEMPxtemptrain = pd.concat([TEMPTraindata,TEMPTraindataxCNDC],axis=1)
TEMPxtemptest = pd.concat([TEMPTestdata,TEMPtestdataxCNDC],axis=1)
for i in range(0,5):
    TEMPdata = TEMPxtemptrain.iloc[:,[i,i+5]]
    TEMPdata = TEMPdata.dropna()
    TEMPtrain = TEMPdata.iloc[:,0]
    temptrain = TEMPdata.iloc[:,[1]]
    
    regressorTEMPxtemp = MLPRegressor()
    regressorTEMPxtemp.fit(temptrain,TEMPtrain)
    
    TEMPdatatest = TEMPxtemptest.iloc[:,[i,i+5]]
    TEMPdatatest = TEMPdatatest.dropna()
    TEMPdatatest = TEMPdatatest.reset_index().iloc[:,1:]
    globals()['TEMPtest'+str(i)]  = TEMPdatatest.iloc[:,0]
    temptest = TEMPdatatest.iloc[:,[1]]
    globals()['predCNDC'+str(i)] = regressorTEMPxtemp.predict(temptest)


predCNDC0 = pd.DataFrame(predCNDC0)
predCNDC0.to_csv("TEMP0CNDC.csv")
predCNDC1 = pd.DataFrame(predCNDC1)
predCNDC1.to_csv("TEMP1CNDC.csv")
predCNDC2 = pd.DataFrame(predCNDC2)
predCNDC2.to_csv("TEMP2CNDC.csv")
predCNDC3 = pd.DataFrame(predCNDC3)
predCNDC3.to_csv("TEMP3CNDC.csv")
predCNDC4 = pd.DataFrame(predCNDC4)
predCNDC4.to_csv("TEMP4CNDC.csv")

TEMPtest0 = pd.DataFrame(TEMPtest0)
TEMPtest0.to_csv("TEMP0truCNDC.csv")
TEMPtest1 = pd.DataFrame(TEMPtest1)
TEMPtest1.to_csv("TEMP1truCNDC.csv")
TEMPtest2 = pd.DataFrame(TEMPtest2)
TEMPtest2.to_csv("TEMP2truCNDC.csv")
TEMPtest3 = pd.DataFrame(TEMPtest3)
TEMPtest3.to_csv("TEMP3truCNDC.csv")
TEMPtest4 = pd.DataFrame(TEMPtest4)
TEMPtest4.to_csv("TEMP4truCNDC.csv")



TEMPxtemptrain = pd.concat([TEMPTraindata,TEMPTraindataxTURB],axis=1)
TEMPxtemptest = pd.concat([TEMPTestdata,TEMPtestdataxTURB],axis=1)
for i in range(0,5):
    TEMPdata = TEMPxtemptrain.iloc[:,[i,i+5]]
    TEMPdata = TEMPdata.dropna()
    TEMPtrain = TEMPdata.iloc[:,0]
    temptrain = TEMPdata.iloc[:,[1]]
    
    regressorTEMPxtemp = MLPRegressor()
    regressorTEMPxtemp.fit(temptrain,TEMPtrain)
    
    TEMPdatatest = TEMPxtemptest.iloc[:,[i,i+5]]
    TEMPdatatest = TEMPdatatest.dropna()
    TEMPdatatest = TEMPdatatest.reset_index().iloc[:,1:]
    globals()['TEMPtest'+str(i)]  = TEMPdatatest.iloc[:,0]
    temptest = TEMPdatatest.iloc[:,[1]]
    globals()['predTURB'+str(i)] = regressorTEMPxtemp.predict(temptest)


predTURB0 = pd.DataFrame(predTURB0)
predTURB0.to_csv("TEMP0TURB.csv")
predTURB1 = pd.DataFrame(predTURB1)
predTURB1.to_csv("TEMP1TURB.csv")
predTURB2 = pd.DataFrame(predTURB2)
predTURB2.to_csv("TEMP2TURB.csv")
predTURB3 = pd.DataFrame(predTURB3)
predTURB3.to_csv("TEMP3TURB.csv")
predTURB4 = pd.DataFrame(predTURB4)
predTURB4.to_csv("TEMP4TURB.csv")

TEMPtest0 = pd.DataFrame(TEMPtest0)
TEMPtest0.to_csv("TEMP0truTURB.csv")
TEMPtest1 = pd.DataFrame(TEMPtest1)
TEMPtest1.to_csv("TEMP1truTURB.csv")
TEMPtest2 = pd.DataFrame(TEMPtest2)
TEMPtest2.to_csv("TEMP2truTURB.csv")
TEMPtest3 = pd.DataFrame(TEMPtest3)
TEMPtest3.to_csv("TEMP3truTURB.csv")
TEMPtest4 = pd.DataFrame(TEMPtest4)
TEMPtest4.to_csv("TEMP4truTURB.csv")


TEMPxtemptrain = pd.concat([TEMPTraindata,TEMPTraindataxCPHL],axis=1)
TEMPxtemptest = pd.concat([TEMPTestdata,TEMPtestdataxCPHL],axis=1)
for i in range(0,5):
    TEMPdata = TEMPxtemptrain.iloc[:,[i,i+5]]
    TEMPdata = TEMPdata.dropna()
    TEMPtrain = TEMPdata.iloc[:,0]
    temptrain = TEMPdata.iloc[:,[1]]
    
    regressorTEMPxtemp = MLPRegressor()
    regressorTEMPxtemp.fit(temptrain,TEMPtrain)
    
    TEMPdatatest = TEMPxtemptest.iloc[:,[i,i+5]]
    TEMPdatatest = TEMPdatatest.dropna()
    TEMPdatatest = TEMPdatatest.reset_index().iloc[:,1:]
    globals()['TEMPtest'+str(i)]  = TEMPdatatest.iloc[:,0]
    temptest = TEMPdatatest.iloc[:,[1]]
    globals()['predCPHL'+str(i)] = regressorTEMPxtemp.predict(temptest)


predCPHL0 = pd.DataFrame(predCPHL0)
predCPHL0.to_csv("TEMP0CPHL.csv")
predCPHL1 = pd.DataFrame(predCPHL1)
predCPHL1.to_csv("TEMP1CPHL.csv")
predCPHL2 = pd.DataFrame(predCPHL2)
predCPHL2.to_csv("TEMP2CPHL.csv")
predCPHL3 = pd.DataFrame(predCPHL3)
predCPHL3.to_csv("TEMP3CPHL.csv")
predCPHL4 = pd.DataFrame(predCPHL4)
predCPHL4.to_csv("TEMP4CPHL.csv")

TEMPtest0 = pd.DataFrame(TEMPtest0)
TEMPtest0.to_csv("TEMP0truCPHL.csv")
TEMPtest1 = pd.DataFrame(TEMPtest1)
TEMPtest1.to_csv("TEMP1truCPHL.csv")
TEMPtest2 = pd.DataFrame(TEMPtest2)
TEMPtest2.to_csv("TEMP2truCPHL.csv")
TEMPtest3 = pd.DataFrame(TEMPtest3)
TEMPtest3.to_csv("TEMP3truCPHL.csv")
TEMPtest4 = pd.DataFrame(TEMPtest4)
TEMPtest4.to_csv("TEMP4truCPHL.csv")



TEMPxtemptrain = pd.concat([TEMPTraindata,TEMPTraindataxPAR],axis=1)
TEMPxtemptest = pd.concat([TEMPTestdata,TEMPtestdataxPAR],axis=1)
for i in range(0,5):
    TEMPdata = TEMPxtemptrain.iloc[:,[i,i+5]]
    TEMPdata = TEMPdata.dropna()
    TEMPtrain = TEMPdata.iloc[:,0]
    temptrain = TEMPdata.iloc[:,[1]]
    
    regressorTEMPxtemp = MLPRegressor()
    regressorTEMPxtemp.fit(temptrain,TEMPtrain)
    
    TEMPdatatest = TEMPxtemptest.iloc[:,[i,i+5]]
    TEMPdatatest = TEMPdatatest.dropna()
    TEMPdatatest = TEMPdatatest.reset_index().iloc[:,1:]
    globals()['TEMPtest'+str(i)]  = TEMPdatatest.iloc[:,0]
    temptest = TEMPdatatest.iloc[:,[1]]
    globals()['predPAR'+str(i)] = regressorTEMPxtemp.predict(temptest)


predPAR0 = pd.DataFrame(predPAR0)
predPAR0.to_csv("TEMP0PAR.csv")
predPAR1 = pd.DataFrame(predPAR1)
predPAR1.to_csv("TEMP1PAR.csv")
predPAR2 = pd.DataFrame(predPAR2)
predPAR2.to_csv("TEMP2PAR.csv")
predPAR3 = pd.DataFrame(predPAR3)
predPAR3.to_csv("TEMP3PAR.csv")
predPAR4 = pd.DataFrame(predPAR4)
predPAR4.to_csv("TEMP4PAR.csv")

TEMPtest0 = pd.DataFrame(TEMPtest0)
TEMPtest0.to_csv("TEMP0truPAR.csv")
TEMPtest1 = pd.DataFrame(TEMPtest1)
TEMPtest1.to_csv("TEMP1truPAR.csv")
TEMPtest2 = pd.DataFrame(TEMPtest2)
TEMPtest2.to_csv("TEMP2truPAR.csv")
TEMPtest3 = pd.DataFrame(TEMPtest3)
TEMPtest3.to_csv("TEMP3truPAR.csv")
TEMPtest4 = pd.DataFrame(TEMPtest4)
TEMPtest4.to_csv("TEMP4truPAR.csv")
