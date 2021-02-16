# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 20:31:57 2021

@author: Charlie
"""
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
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
from sklearn.model_selection import train_test_split
 
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

def data(tag):
    data = stock(tag)
    history_points = 500
   
    #x = np.array([data[i  : i + history_points].copy() for i in range(len(data) - history_points)])
    x = data.iloc[0:(5000 - history_points), 0]
    x = x.to_numpy()
    x = x.reshape(history_points, -1)
    
    y = (data.iloc[(5000 - history_points):5000, 0])
    
  
    
    print(x.shape, y.shape)

    return x, y

def model(tag):
    X, y = data(tag)
    data_dmatrix = xgb.DMatrix(data=X, label=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    xg_reg = XGBRegressor(max_depth=5, learning_rate=0.2, n_estimators=50, n_jobs=-1, colsample_bytree=0.1)
    xg_reg.fit(X_train, y_train)
    preds = xg_reg.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))

    y_test = list(y_test)
    
    for i in range(len(y_test)):
        y_test[i] = float(y_test[i])
    
    #print(y_test.shape)
    print(y_test)
    print(preds)
    
    
    x_ax = range(50)
    plt.plot(x_ax, y_test, label = "origianal")
    plt.plot(x_ax, preds, label = "prediction")
    plt.legend()
    plt.show()
    
    return

print(model('AAPL'))
