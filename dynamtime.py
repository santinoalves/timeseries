# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 23:45:56 2020

@author: Connor Simpson
"""
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
from fastdtw import fastdtw

bioChemAll42 = pd.read_csv("bioChemAll43.csv")
bioChemAll42= bioChemAll42.iloc[:,1:]

dis2 = []
for j in range(0,len(bioChemAll42.columns)):
    x = bioChemAll42["darwinBiochem_TEMP_0"]
    x = x[~np.isnan(x)]
    x2 = (x-min(x))/(max(x)-min(x))
    y = bioChemAll42[bioChemAll42.columns[j]]
    y = y[~np.isnan(y)]
    y2 = (y-min(y))/(max(y)-min(y))
    distance, path = fastdtw(x2, y2, dist=euclidean)
    dis2.append(distance)


askk2 = pd.DataFrame(dis2,index=bioChemAll42.columns)
askk2sort = askk2.sort_values(askk2.columns[0])

