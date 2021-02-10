# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 23:55:53 2020

@author: rajee
"""
import pandas as pd
from hotelling.plots import control_chart, control_stats, univariate_control_chart

dataset = pd.read_csv('IDV4.csv')
X = dataset.iloc[:, 0:21]



from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 7, kernel = 'rbf')
    
X_test = kpca.fit_transform(X)

X_test = pd.DataFrame(X_test)
univariate_control_chart(dataset.iloc[:,1], legend_right=True, interactive=True);

control_chart(X_test, phase=1, alpha=0.1, x_bar=None, s=None, legend_right=False, interactive=False, width=10)

