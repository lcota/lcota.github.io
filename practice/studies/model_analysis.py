# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:33:46 2024

@author: lcota
"""
#%% imports
from importlib import reload

import datatable as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import statsmodels.api as sm

pd.options.display

import funcs
from funcs import *
import datatools
from datatools import *
import strategy
from strategy import *
import features
from features import dfx, dfy, dfref



#%% load data
results = pd.read_parquet("data/results/results.pq")
dftr = pd.read_parquet("data/workspace/dftotrtn.pq").reset_index()
dfref = pd.read_csv("data/processed/etfref.csv")

dftr['index'] = dftr['index'].apply(lambda x: pd.to_datetime(x).date())

dftr.set_index('index', inplace=True)
tickers = results['ticker'].unique()

#%% calc returns on pmi pub dates

# adjust trading dates for NMI/PMI release dates
cols = ['cpipubdate', 'nmipubdate', 'pmipubdate']
dfdates = dfx[cols].reset_index().copy()

# base buy dates on pmi date for same month
# adj_date(x, ndays=3, force=False, mod_backward=False, country='USA'):
dfdates['buydate'] = dfdates['pmipubdate'].apply(lambda x: adj_date(x, ndays=1))
dfdates['selldate'] = dfdates['buydate'].apply(lambda x: adj_date(x, ndays=16, force=True))

buydates = dfdates['buydate'].values
selldates = dfdates['selldate'].values
buydates = [d for d in buydates if (d >= dftr.index[0]) and (d <= dftr.index[-1])]
selldates = [d for d in selldates if (d >= dftr.index[0]) and (d <= dftr.index[-1])]

#%% Load predictions & calc returns per etf at each ISM date, not CPI dates
dfmodels = pd.read_parquet("data/results/results_ex_cpi.pq")
dfmodels['ticker'] = dfmodels['ticker'].apply(lambda x: x.upper())


dfbuys = dftr.loc[buydates]
dfsells = dftr.loc[selldates]

dfbuys.to_parquet("data/results/buys_ism_dates.pq")
dfsells.to_parquet("data/results/sells_ism_dates.pq")

# rtns = (selllevels / buylevels) - 1.0
index = dfbuys.index
index = pd.Series(index).apply(lambda x :x.strftime("%b.%Y").upper())

rtn_ism = pd.DataFrame(data=(dfsells.values / dfbuys.values) - 1, columns=dfbuys.columns)
rtn_ism['buydate'] = pd.Series(dfbuys.index.values)
rtn_ism['selldate'] = pd.Series(dfsells.index.values)

rtn_ism['refmonthyear'] = rtn_ism['buydate'].apply(lambda x :x.strftime("%b.%Y").upper())
rtn_ism.set_index('refmonthyear', inplace=True)

#%% Total returns @ ISM dates, using model prediction
def tr(dfrtn, dfmdl):
    rtns = dfrtn[ticker]
    dfsig = dfmdl[dfmdl['ticker'] == ticker]
    return 100*(closes.iloc[-1] / closes.iloc[0] - 1)

# df[base_trcol] = (1 + df[returns.name]).cumprod() * 100
# dftr_ism = (1 + rtn_ism).cumprod() * 100

lsrtns = {}
lortns = {}
ticker = 'BND'
for ticker in dftr_ism.columns:
    sig = dfmodels[dfmodels['ticker'] == ticker]
    
    df = pd.DataFrame(rtn_ism[ticker].copy())
    df['sig.ls'] = sig['ls']
    df['sig.lo'] = sig['lo']
    df = df.dropna()
    df['rtn.ls'] = (1 + (df['sig.ls'] * df[ticker])).cumprod() * 100
    df['rtn.lo'] = (1 + (df['sig.lo'] * df[ticker])).cumprod() * 100
    
    lsrtns[ticker] = df['rtn.ls']
    lortns[ticker] = df['rtn.lo']



lsrtns = pd.DataFrame(lsrtns)
lortns = pd.DataFrame(lortns)

lsrtns.tail(1).to_clipboard()
lortns.tail(1).to_clipboard()
lortns

#%%
head(dftr_ism)
tail(dftr_ism)

head(dftr_ism, 1)
tail(dftr_ism, 1)

dftr_ism.iloc[-1].to_clipboard()
dftr_ism.iloc[0].to_clipboard()
bnd
aggreturn = (dftr_ism.tail(1).dropna() / dftr_ism.head(1).dropna()) - 1.0
aggreturn

#%% 
trr = {}
# df.loc[dates] 
# for ticker in tickers:
ticker = 'SPY'
idx = closes.index
head(closes)
tail(closes)

dfmodels


