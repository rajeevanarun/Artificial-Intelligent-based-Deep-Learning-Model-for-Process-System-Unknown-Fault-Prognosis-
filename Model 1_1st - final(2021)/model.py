# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 02:39:26 2020

@author: rajee
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LSTM_model import model_forecast


def model_LSTM_CNN(kp, PC, Y1, series, window_size):
    tf.keras.backend.clear_session
    tf.random.set_seed(51)
    np.random.seed(51)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=8, kernel_size=13,
                strides=1, padding="causal",
                activation="relu",
                input_shape=[None, kp]),
#    tf.keras.layers.MaxPooling1D(pool_size = 2, strides=2, padding='valid'),
#    tf.keras.layers.Flatten(),
    tf.keras.layers.LSTM(10, return_sequences=True),
    tf.keras.layers.LSTM(5, return_sequences=True),    
#    tf.keras.layers.LSTM(2, return_sequences=True),
#    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(6, activation="relu"),
#    tf.keras.layers.Dense(2, activation="relu"),
    tf.keras.layers.Dense(kp)])

    optimizer = tf.keras.optimizers.SGD(lr=1e-30, momentum=1)

    model.compile(loss=tf.keras.losses.Huber(),
                   optimizer='adam',
                   metrics=["mse"])

    model.fit(PC, Y1, epochs=200)

    rnn_forecast = model_forecast(model, series, window_size)
    rnn_forecast = np.reshape(rnn_forecast, (100,kp))
    return rnn_forecast