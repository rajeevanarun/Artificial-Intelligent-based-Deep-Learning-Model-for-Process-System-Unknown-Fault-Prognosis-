# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:16:30 2020

@author: rajee
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"Plot Series"
def plot_series(time, series, label, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format, label = label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
import csv
time_step = []
data = []

from KPCA import KPCA_M
[X] = KPCA_M()
series = X
time_step = range(0,100)


#Plot Data
series = np.array(series)
time = np.array(time_step)
plt.figure(figsize=(12, 8))
plot_series(time, series,"ser")

#Split train and valid data
split_time = 100
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

series = np.reshape(series,(10,10,6))

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series) # Tesnsor input data
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(10).prefetch(1)
    forecast = model.predict(ds)
    return forecast

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

#Data Window
window_size = 50
batch_size = 80
shuffle_buffer_size = 100

# learning
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(train_set)
print(x_train.shape)

# model for learning 

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=20, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=(None, 1)),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.Dense(20, activation="relu"),
  tf.keras.layers.Dense(40, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 250)])


optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.7)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer='adam',
              metrics=["mae"])

history = model.fit(train_set,epochs=75)

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, "real" )
plot_series(time_valid, rnn_forecast, "predict")
plt.legend(loc="RP")
plt.title("IDV_15 Fault prediction")
tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
