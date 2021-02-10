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


def model_LSTM_CNN(kp, x_test, y_test, y_hat, series, window_size):
    tf.keras.backend.clear_session
    tf.random.set_seed(51)
    np.random.seed(51)

    model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(10, return_sequences=True),
    tf.keras.layers.LSTM(6, return_sequences=True),    
    # tf.keras.layers.LSTM(2, return_sequences=True),
#    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(2, activation="relu"),
#    tf.keras.layers.Dense(2, activation="relu"),
    tf.keras.layers.Dense(kp)])
    
    model.compile(loss=tf.keras.losses.Huber(),
                   optimizer='adam',
                   metrics=["mse"])

    model.fit(x_test, y_test, epochs=200)

    rnn_forecast = model_forecast(model, series, window_size)
    
    # model2 = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv1D(filters=8, kernel_size=13,
    #             strides=1, padding="valid",
    #             activation="relu",
    #             input_shape=[None, kp]),
    #     tf.keras.layers.MaxPooling1D(pool_size = 2,strides = None, padding='valid'),
    #     # tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(200,activation='relu'),
    #     tf.keras.layers.Dense(200,activation='softmax')]
    #     )

    # optimizer = tf.keras.optimizers.SGD(lr=1e-30, momentum=1)

    # model2.compile(loss=tf.keras.losses.Huber(),
    #                 optimizer='adam',
    #                 metrics=["mse"])

    # model2.fit(x_test, y_hat,  epochs=200)

    return rnn_forecast