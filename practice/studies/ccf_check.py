# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:13:42 2023

@author: lcota
"""
#%% imports 
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as ts
import matplotlib.pyplot as plt
import seaborn as sb

import numpy as np
import scipy

from pandas import Series, DataFrame

#%% utility functions - lags
def lag(x, nlags):
    return x.shift(-nlags)

def lag_mat(x, nlags):
    df = DataFrame(x)
    df.columns = ['x_lag0']
    for i in range(1, nlags+1):
        cname = f"x_lag{i}"
        df[cname] = x.shift(i)
        
    return df

def ccf(x, y, nlags):
    X = lag_mat(x, nlags)
    
    
    return df


#%% load cpi & returns
dfcpi = pd.read_feather("data/processed/cpi_diffs.fthr")
dfreturns = pd.read_feather("data/processed/monthly_trade_returns.fthr")

#%% checking cross correlation function
cpi_mom = dfcpi['cpi.mom.diff']
cpi_yoy = dfcpi['cpi.yoy.diff']


#%% toy time series for ccf test

# x = np.arange(0, 20, .1)
x = np.arange(0, 1000)
ret = np.random.normal(.05, scale=.5, size=len(x))/100.
y = np.cumprod(1 + ret)

# plt.plot(y)

#%% generate 2nd time series 
y2 = np.cumprod(1 + np.random.normal(0.1, scale=.3, size=len(x))/100.)
# plt.plot(y2)

#%% check CCF of TLT, MOM YoY
# ts.ccf(y, y2, nlags=12)
Y = dfreturns['TMV']


plt.scatter(cpi_yoy, Y)
plt.scatter(cpi_mom, Y)

xc = plt.xcorr(y, y2, maxlags=12)