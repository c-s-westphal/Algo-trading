# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 19:45:03 2021

@author: Charlie
"""
from alpha_vantage.timeseries import TimeSeries 
from alpha_vantage.techindicators import TechIndicators 
import pandas as pd
import time

empty = pd.DataFrame()

def download_stock(tag):
  key = '5DBJSH6SQWHZD5E9'
  ts = TimeSeries(key)
  stock, meta = ts.get_daily(symbol=tag, outputsize='full')
  return stock, meta

#converting array to DataFrame
def stock(tag, df_to_add, date_from, date_to):
  df_, _ = download_stock(tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[dates]
  df_ = df_['1. open']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def get_index(tag, date_from, date_to):
  df_, _ = download_stock(tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['1. open']
  df_ = pd.DataFrame(df_)
  #extract index and return it as series
  df_ = df_.reset_index()
  return df_['index']

def stock_sma(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_sma(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['SMA']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_ema(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_ema(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['EMA']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_wma(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_wma(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['WMA']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_dema(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_dema(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['DEMA']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_tema(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_tema(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['tema']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_trima(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_trima(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['TRIMA']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_kama(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_kama(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['MAMA']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_vwap(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_vwap(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['VWAP']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_t3(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_t3(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['T3']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_macd(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_macd(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['MACD']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_macdext(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_macdext(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['MACDEXT']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_stoch(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_stoch(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['STOCH']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_stochf(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_stochf(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['STOCHF']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_rsi(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_rsi(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['RSI']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_stochrsi(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_stochrsi(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['STOCHRSI']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_willr(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_willr(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['WILLR']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_adx(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_adx(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['ADX']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_adxr(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_adxr(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['ADXR']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_apo(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_apo(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['APO']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_ppo(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_ppo(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['PPO']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_mom(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_mom(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['MOM']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_bop(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_bop(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['BOP']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_cci(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_cci(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['CCI']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_cmo(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_cmo(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['CMO']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_aroonosc(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_aroonosc(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['AROONOSC']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_roc(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_roc(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['ROC']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_rocr(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_rocr(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['ROCR']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_aroon(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_aroon(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['AROON']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_mfi(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_mfi(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['MFI']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_trix(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_trix(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['TRIX']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_ultosc(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_ultosc(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['ULTOSC']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_dx(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_dx(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['DX']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_minus_di(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_minus_di(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['MINUS_DI']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_plus_di(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_plus_di(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['PLUS_DI']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_minus_dm(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_minus_dm(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['MINUS_DM']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_plus_dm(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_plus_dm(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['PLUS_DM']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_bbands(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_bbands(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['BBANDS']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_midpoint(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_midpoint(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['MIDPOINT']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_midprice(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_midprice(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['MIDPRICE']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_sar(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_sar(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['SAR']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_trange(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_trange(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['TRANGE']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_atr(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_atr(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['ATR']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_natr(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_natr(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['NATR']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_ad(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_ad(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['AD']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_adosc(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_adosc(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['ADOSC']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_obv(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_obv(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['OBV']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_ht_trendline(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_ht_trendline(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['HT_TRENDLINE']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_ht_sine(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_ht_sine(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['HT_SINE']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_ht_trendmode(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_ht_trendmode(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['HT_TRENDMODE']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_ht_dcperiod(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_ht_dcperiod(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['HT_DCPERIOD']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_ht_dcphase(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_ht_dcphase(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['HT_DCPHASE']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_

def stock_ht_phasor(tag, df_to_add, date_from, date_to):
  key = '5DBJSH6SQWHZD5E9'
  ti = TechIndicators(key)
  df_, _ = ti.get_ht_phasor(symbol=tag)
  df_ = pd.DataFrame(df_)
  df_ = df_.transpose()
  df_ = df_.loc[date_from : date_to]
  df_ = df_['IBM']
  df_ = pd.DataFrame(df_)
  df_.columns = [tag]
  df_ = df_[tag].values.astype(float)
  df_ = pd.concat([df_to_add, pd.DataFrame(df_)], axis=1)
  return df_


def combiner(tag, func, date_from, date_to):
    ind=get_index(tag, date_from, date_to)
    df=func(tag, ind, date_from, date_to)
    
    return df

print(combiner('AAPL', stock_ht_phasor,'2020–05–22','2019–04–22'))
#index
#csv of stocks



  