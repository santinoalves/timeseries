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
