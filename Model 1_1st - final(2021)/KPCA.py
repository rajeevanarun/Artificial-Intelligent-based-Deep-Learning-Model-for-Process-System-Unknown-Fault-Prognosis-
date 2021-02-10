# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 01:00:58 2020

@author: rajee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hotelling.stats import hotelling_t2
from sklearn.svm import OneClassSVM
from sklearn.decomposition import KernelPCA
from sklearn import preprocessing

def KPCA_M(dataset, st, window_size,n):
#    dataset = pd.DataFrame(dataset)
    d = preprocessing.scale(dataset)
    X_test = d[st: st+window_size, 0:21]
    Y_test = d[st+n:st+window_size+n, 0:21]
    X_t = d[0:600, 0:21]
    y_t = d[0:300, 22]
    anomaly_count = []
    
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_test = sc.transform(X)

#     kpca = KernelPCA(n_components = 6, kernel = 'linear')
# ##
#     X_t = kpca.fit_transform(X_t)    
#     X_test = kpca.transform(X_test)
#     Y_test = kpca.transform(Y_test)
#    X_test = X_test.T
#    Y_test = Y_test.T
 #   res = hotelling_t2(X_test, Y_test[0:2,0:6])
    
#    from sklearn.neighbors import KNeighborsRegressor
#    neigh = KNeighborsRegressor(n_neighbors=50)
#    model = neigh.fit(X_t, y_t)
#    X_test = model.predict(X_test)
#    Y_test = model.predict(Y_test)
#    
 #   y_predict = y_predict.tolist()
#    y_predict.append(time_tr)
    
 #   plt.plot(y_predict)
    ee = OneClassSVM(gamma = 'auto', nu=0.0008, kernel = 'rbf').fit(X_t)
    for st in range (0, len(X_t),100):
        yh = ee.predict(X_t[st:(st+100),0:21])
        count = np.count_nonzero(yh == -1)
        anomaly_count.append(count)
    margin = 2* (max(anomaly_count))
    return [X_test,Y_test, ee,margin]