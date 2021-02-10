# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:01:23 2020

@author: rajee
"""
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from hotelling.stats import hotelling_t2
from hotelling.plots import control_chart, control_stats, univariate_control_chart
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

y = pd.read_csv('IDV6.csv')
X = y.iloc[0: 1500, 1:22]
y_t = y.iloc[0:1500, 23]
#y = y.iloc[1400:1500, 0:21]
kpca = KernelPCA(n_components = 6, kernel = 'rbf')
X_test = kpca.fit_transform(X)
X_test = pd.DataFrame(data=X_test)

y_test = kpca.fit_transform(y)
y_test = pd.DataFrame(data=y_test)

X_t = X_test[0:600]
X_te = X_test[700:800]
X_te2 = X_test[1100:1200]

ee = OneClassSVM(kernel = 'linear', gamma=0.01, nu = 0.5).fit(X_t)
yhat1 = ee.predict(X_te)
yhat2 = ee.predict(X_te2)

#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(X_t, y_t)


#yhat1 = neigh.predict(X_te)
#yhat2 = neigh.predict(X_te2)

print (np.count_nonzero(yhat1 == -1))
print (np.count_nonzero(yhat2 == -1))

#res = hotelling_t2(X_test, y_test, bessel=True, S=None)
#print (res[0])
##
#print(res)


#control_chart(X_test, alpha = 0.06,phase = 2)