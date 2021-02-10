# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 23:21:54 2020

@author: rajee
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

def get_data():
    dataset = pd.read_csv('IDV15.csv')
    
# preprocessing
    #x1 = preprocessing.normalize(x, norm='l2')
    x1 = preprocessing.scale(x)             # scale the data
                           # train Data
    x2 = x1[0:300:] #400:601
    x2in1 = x1[1150:1450,:] 
    x2 = np.concatenate((x2, x2in1), axis=0)
    x2in2 = x1[1650:1950,:]
    x2 = np.concatenate((x2, x2in2), axis=0)
    x2in3 = x1[2150:2450,:]
    x2 = np.concatenate((x2, x2in3), axis=0)
    
#    x2 = x1[1:100,:]    # 1:600
    x3 = x1[0:3001,:]
#     
    
# Apply PCA
#    pca = PCA(n_components=4)
#    pca.fit(x1)
#    x_pca = pca.transform(x1)
#    x2 = x_pca[0:2500,:]                       # train Data
#    x3 = x_pca[0:3000,:]                    # test data
 #   y = np.array(df['label'])
#    y1 = y[1:600]
#    y2 = y[1:1500]
#    x_train,x_test,y_train,y_test = train_test_split(x1,y,test_size=0.2)  
    x_train = x2 
    x_test = x3
 #   y_train = y1
 #   y_test = y2
    return[x,x_train,x_test]