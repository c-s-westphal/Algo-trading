# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 19:25:06 2021

@author: Charlie
"""

import numpy as np #The Numpy numerical computing library
import pandas as pd #The Pandas data science library
import requests #The requests library for HTTP requests in Python
import xlsxwriter #The XlsxWriter libarary for 
import math #The Python math module
from scipy import stats #The SciPy stats module

#codes = ['FOUR','ASL','AGK','AAF','ATST','ATT','APAX','ASCL','ASHM','AGR','AML','AGT','AVON','BAB','BGFD','BGS','USA','BBY','BNKR','BBGI','BBH','BEZ','AJB','BWY','BIFF',
 #        'BRSC','BRWM','BCPT','BGSC','BOY','BRW','BVIC','CCR','CNE','CLDN','CLSN','CPI','CAPC','CCL','CEY','CNA','CHG']

codes = ['FOUR','USA','AAPL','MSFT','CSCO','INTC','PLTR']

def data_collector(code):

    data = []

    for i in range(len(code)):
        data.append(pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/'+ code +'?period1=1579807912&period2=1611430312&interval=1d&events=history&includeAdjustedClose=true'))

    data = pd.DataFrame(data)

    return data

#print(data[0][1]['Close'])
#jul = [1,2,3,4,5,6]
#print(jul)

def avg_daily_grwth(codes):
    
    data = data_collector(codes)

    
    daily_growth = []
    
    for i in range(len(data)-1):
        if len(data[0][i]['Open']) <= 261:
            for j in range(len(data[0][i]['Open'])):
                daily_growth.append(((data[0][i]['Close'][j]-data[0][i]['Open'][j])*100)/data[0][i]['Open'][j])
        else:
            for j in range(261):
                daily_growth.append(((data[0][i]['Close'][j]-data[0][i]['Open'][j])*100)/data[0][i]['Open'][j])
            
        
    avg_daily_growth = np.mean(daily_growth)
    
    return avg_daily_growth

def growth_compiler(codes):
    
    growths = []
    for i in range(len(codes)):
        day_growth = avg_daily_grwth(codes[i])
        growths.append(day_growth)
        
    growths.append(codes)
    
    return growths

k=growth_compiler(codes)
print(k)
        