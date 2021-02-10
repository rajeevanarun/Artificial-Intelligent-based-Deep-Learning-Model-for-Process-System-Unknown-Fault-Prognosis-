# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:53:17 2020

@author: rajee
"""

import numpy as np
import tensorflow as tf
import tflearn
import tflearn.variables as va
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as srn
from OC_NN import One_Class_NN_explicit_linear,One_Class_NN_explicit_sigmoid
from tf_OneClass_LSTM_AE_NN_sigmoid import tf_OneClass_LSTM_AE_NN_sigmoid
from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression, oneClassNN

from de import get_data

[data,data_train,data_test] = get_data()

#data_train = data[0:220]
#target = Xlabels
X = data_train
#Y = pd.get_dummies(targets_train)       # conver to one-hot

#Y = Y.tolist()
#Y = [[i] for i in Y]
#
# For testing the algorithm
X_test = data_test
#Y_test = targets_tes
#Y_test = Y_test.tolist()
#Y_test = [[i] for i in Y_test]
D_scores = {}

#result = One_Class_NN_explicit_linear(data_train,data_test)
#D_scores["One_Class_NN_explicit-Linear-Train"] = result[0]
#D_scores["One_Class_NN_explicit-Linear-Test"] =  result[1]
#from dic_onc import tf_OneClass_NN_linear
#result = tf_OneClass_NN_linear(data_train,data_test)
#D_scores["tf_OneClass_NN_linear"] = result[0]
#D_scores["tf_OneClass_NN_linear"] =  result[1]
#
test_con = 0


    


from dic_onc import tf_OneClass_NN_sigmoid
result = tf_OneClass_NN_sigmoid(data_train,data_test)
D_scores["tf_OneClass_NN_sigmoid"] = result[0]
D_scores["tf_OneClass_NN_sigmoid"] =  result[1]


#
#from dic_onc import tf_OneClass_NN_Relu
#result = tf_OneClass_NN_Relu(data_train,data_test)
#D_scores["tf_OneClass_NN_Relu"] = result[0]
#D_scores["tf_OneClass_NN_Relu"] =  result[1]

#anomaly_linear_train = sum(n < 0 for n in result[0])
#anomaly_linear_test = sum(n < 0 for n in result[1])
#
#result1 = One_lass_NN_explicit_sigmoid(data_train,data_test)0ss_NN_explicit-Sigmoid-Train"] = result1[0]
#D_scores["One_Class_NN_explicit-Sigmoid-Test"] = result1[1]
#anomaly_sigmoid_train = sum(n < 0 for n in result1[0])
#anomaly_sigmoid_test = sum(n < 0 for n in result1[1])

#result1 = tf_OneClass_LSTM_AE_NN_sigmoid(data_train,data_test,0.02) # 0.02 for good result
#D_scores["tf_OneClass_LSTM_AE_NN_sigmoid_train"] = result1[0]
#D_scores["tf_OneClass_LSTM_AE_NN_sigmoid_test"] = result1[1]

#anomaly_sigmoid_train = sum(n < 0 for n in result1[0])
#anomaly_sigmoid_test = sum(n < 0 for n in result1[1])
#
#from plot_NN import plot_NN
#plot_NN(data,D_scores)

test_result = result[1]
train_result = result[0]
le = len(train_result)

# Algorithm - count the anomalies/
anomaly_points = 0
ap_arr = []
ap_arr_test = []
window_size = 100
length = len(test_result)
#for i in test_result[1+window : 100+window]:
#    if i<0:
#        anomaly_points = anomaly_points+1
#    else:
#        anomaly_points = anomaly_points

for i in range(1,length,window_size):
    anomaly_points = sum(n<0 for n in test_result[i:i+window_size])
    ap_arr.append(anomaly_points)

# Margin setup
for i1 in range(1,le,window_size):
    anomaly_points_test = sum(n1<0 for n1 in train_result[i1:i1+window_size])
    ap_arr_test.append(anomaly_points_test)
    
margin = np.max(np.asarray(ap_arr_test, dtype = int))

 
plt.plot(ap_arr)
plt.ylabel('no_of_anomalies')
plt.xlabel('samples (100 windowsize)')
plt.title('Tenesse Eastman Process Fault condition')
plt.ylim(1,window_size)
plt.xlim(1,(length/window_size))
plt.grid(True)
plt.axhline(y=margin, color='r', linestyle='--')
plt.show()