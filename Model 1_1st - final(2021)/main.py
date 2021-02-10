
 # -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 02:45:30 2020

@author: rajee
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hotelling.plots import control_chart, control_stats, univariate_control_chart
from hotelling.stats import hotelling_t2
from sklearn.svm import OneClassSVM

from KPCA import KPCA_M
from LSTM_model import model_forecast
from model import model_LSTM_CNN

def plot_series(time, series, label, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format, label = label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

dataset = pd.read_csv('IDV4.csv')
length = len(dataset)

window_size = 100
batch_size = 25
shuffle_buffer_size = 50
n = 100
st = 0
pc_ser = []
yhat=[]
anomaly_count = [] 
anomaly_count1 = [] 
rnn_f = []
kp = 21

for st in range (0,(length-100), window_size):
   # print(st)
    [PC,Y,ee,margin] = KPCA_M(dataset, st, window_size,n)
   # print(PC)    

    series = np.array(PC)
    pc_ser.append(series)
#    pc_ser = np.array(pc_ser)


    Y = np.array(Y)
    pc_ser_f = np.array(pc_ser, dtype = np.float32)
#    Y1 = np.reshape(Y,(1,100,1))
#    PC = np.reshape(PC,(1,600,1))
    
    Y1 = np.expand_dims(Y, axis=1)
    PC = np.expand_dims(PC, axis=1)
    pc_ser_ff = np.reshape(pc_ser_f,((st+100),kp))
#
#
    rnn_forecast = model_LSTM_CNN(kp, PC, Y1, series, window_size)
    rnn_f.append(rnn_forecast)
    rnn_ff = np.array(rnn_f)
    rnn_ff = np.reshape(rnn_ff,((st+100),kp))
    
    yhat = ee.predict(Y)
    yhat = pd.DataFrame(yhat)
    
    count = np.count_nonzero(yhat == -1)
    anomaly_count.append(count)
    
    yh = ee.predict(rnn_forecast)
    yh = pd.DataFrame(yh)
    
    count1 = np.count_nonzero(yh == -1)
    anomaly_count1.append(count1)
    # yh = np.array(yh)
    # yhat.append(yh)
#
## plot
# #
    time_step = range(0,length)
    time = np.array(time_step)
    t1 = time[0:st+n]
    t2 = time[st+n:st+n+window_size]
###
    y_plot = np.reshape(Y,(100,kp))
    plt.figure(figsize=(10, 6))
    plot_series(t1, pc_ser_ff[:,0], "real" )
    plot_series(t1, rnn_ff[:,0], "predict")
#    plot_series(t2, y_plot, "real")
      # plt.legend(loc="RP")
    plt.title("IDV_1 Fault prediction")
    


mae = tf.keras.metrics.mean_absolute_error(pc_ser_ff,rnn_ff).numpy() 
    
plt.twinx()
plt.figure(figsize=(6,4), dpi=70)
plt.xlabel("Data window")
plt.ylabel("Data anomaly points ")
plt.plot(anomaly_count,"--", color = "g", label = "real")
plt.legend()
plt.plot(anomaly_count1, color ="r", label = "forcast")
plt.title("IDV_4 Fault prediction")
plt.legend()
plt.axhline(y=margin,color='b', linestyle = '--', label= 'risk margin')
plt.show()

#rnn_forecast = pd.DataFrame(rnn_forecast)
#control_chart(rnn_forecast)

#yhat = np.reshape(yhat(1500,1))
