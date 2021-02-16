# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:50:49 2021

@author: Charlie
"""

from alpha_vantage.timeseries import TimeSeries 
from alpha_vantage.techindicators import TechIndicators 
import pandas as pd
import numpy as np
import time
import json
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from keras.utils import plot_model
import matplotlib.pyplot as plt
 
np.random.seed(4)
tf.random.set_seed(4)

empty = pd.DataFrame()

def download_stock(tag):
  key = '5DBJSH6SQWHZD5E9'
  ts = TimeSeries(key)
  stock, meta = ts.get_daily(symbol=tag, outputsize='full')
  return stock, meta

#converting array to DataFrame
def stock(tag):
  df_, _ = download_stock(tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc['2020–05–22':'1999–04–22']
  arr = df_.values.copy()
  arr.resize(5250, 5)
  df_ = pd.DataFrame(arr)
  Open = df_[0]
  low = df_[1]
  close = df_[2]
  high = df_[3]
  
  return df_

def normaliser(stock):
    scaler = MinMaxScaler()
    return scaler.fit_transform(stock)

# using the last {history_points} open high low close volume data points, predict the next open value
def cleaner(tag):
    history_points = 50

    data_normaliser = MinMaxScaler()

    data = stock(tag).to_numpy()
    
    data_normalised = data_normaliser.fit_transform(data)
    
    

    ohlcv_histories_normalised =      np.array([data_normalised[i  : i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array([data_normalised[:,0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
    print(ohlcv_histories_normalised.shape)
    next_day_open_values = np.array([data[i + history_points].copy() for i in range(len(data) - history_points)])
    
    next_day_open_values = np.expand_dims(next_day_open_values_normalised, -1)
    next_day_open_values = np.squeeze(next_day_open_values, 1)

    
    print(data[0:5200,0].shape)
    k = np.expand_dims(data[0:5200,0], -1)
   
    
    print(k.shape)
    
    y_normaliser = MinMaxScaler()
    y_normaliser.fit(( k ))
    
    print(next_day_open_values)
    
    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]
    return ohlcv_histories_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

def set_split(tag):
    ohlcv_histories, next_day_open_values, unscaled_y, y_scaler = cleaner(tag)

    test_split = 0.9 # the percent of data to be used for testing
    n = int(ohlcv_histories.shape[0] * test_split)

    # splitting the dataset up into train and test sets

    ohlcv_train = ohlcv_histories[:n]
    y_train = next_day_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    y_test = next_day_open_values[n:]

    unscaled_y_test = unscaled_y[n:]
    
    return ohlcv_train, y_train, ohlcv_test, y_test, unscaled_y_test, y_scaler

def LSTM_fun(tag):
    
    history_points = 50
    
    ohlcv_histories, y_train, ohlcv_test, y_test, unscaled_y_test, y_scaler = set_split(tag)

    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    model = Model(inputs=lstm_input, outputs=output)

    adam = optimizers.Adam(lr=0.0005)

    model.compile(optimizer=adam, loss='mse')


    plot_model(model, to_file='model.png')
    
    model.fit(x=ohlcv_histories, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    evaluation = model.evaluate(ohlcv_test, y_test)
    print(evaluation)
    
    y_test_predicted = model.predict(ohlcv_test)
    # model.predict returns normalised values
    # now we scale them back up using the y_scaler from before
    
    
    y_test_predicted = y_scaler.inverse_transform(y_test_predicted)

# also getting predictions for the entire dataset, just to see how it performs
    
    y_predicted = model.predict(ohlcv_histories)
    
    y_predicted = y_scaler.inverse_transform(y_predicted)
    
    
    print(unscaled_y_test)
    
    unscaled_y_test = y_scaler.inverse_transform(unscaled_y_test)
    
    #print(unscaled_y_test)
    #assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)

    start = 0
    end = -1
    
    
    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()
    
    
    return 'done' 

LSTM_fun('PLTR')
