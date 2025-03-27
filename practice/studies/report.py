# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:41:08 2024

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

etfs2keep = ['AGG','VCIT','SPIB','BSV','JNK','TBT','TMV','VCLT','LQD','TMF','SPY','SPYG','SPLG','KBE','XRT','DBC','GSG','UNG','XOP', '6040', 'NAIVEVOL']
dfy.columns = [c.upper() for c in dfy.columns]
dfy = dfy[['AGG','VCIT','SPIB','BSV','JNK','TBT','TMV','VCLT','LQD','TMF','SPY','SPYG','SPLG','KBE','XRT','DBC','GSG','UNG','XOP', '6040', 'NAIVEVOL']]
dftr = dftr[etfs2keep]

#%%

#%% 6040 / Naive Vol Total Return
dfy = (1 + dfy.dropna()).cumprod()*100
dfy

head(results)
results[results['base_totrtn'].isna()==False]
#%% Quadrant Plots


dfy_trans = dfy.copy() # transpose of returns matrix, dfy
dfy_trans.columns = [c.upper() for c in dfy_trans.columns]
dfy_trans = dfy_trans[etfs2keep]
dfy_trans = dfy_trans.T.reset_index()
dfy_trans['index'] = dfy_trans['index'].apply(lambda x: x.upper())
dfy_trans['ticker'] = dfy_trans['index'].copy()
dfy_trans.drop('index', axis=1, inplace=True)

# dfy_trans['keep'] = dfy_trans.ticker.apply(lambda x: x in etfs2keep)
# dfy_trans = dfy_trans[dfy_trans.keep == True].copy()
# dfy_trans.drop('keep', axis=1, inplace=True)

# add ref data (category) of fund
dfy_trans = pd.merge(dfy_trans, dfref.groupby('ticker')['group'].last(), left_on='ticker', right_on='ticker', suffixes=('', ''))

# dfy_trans.set_index('index', inplace=True)
# rtnbygrp = dfy_trans.drop(['ticker', 'group_y'], axis=1).groupby(['group'])
rtnbygrp = dfy_trans.groupby(['group'])
rtnbygrp = rtnbygrp.agg('mean').T
rtnbygrp.columns = ['Commodity', 'Debt', 'Equity']
rtnbygrp = pd.merge(dfx['cpiyoy1m.above'], rtnbygroup, left_index=True, right_index=True)


dfall = pd.merge(dfx, dfy_trans, left_index=True, right_index=True, how='left')
dfall.to_clipboard()

#%% melt dfy and dfx
dfplot = pd.merge(left=dfx, right=dfy, left_index=True, right_index=True, how='inner')
dfplot.to_clipboard()


#%% plot matrix
dfplot = pd.melt(dfy_trans, var_name='ticker', ignore_index=False)

dfplot['group'] = dfplot['ticker'].apply(lambda x: dfref.loc[x]['group'])
dfplot

dfplot['cpiyoy1m.above'] = dfx['cpiyoy1m.above']
dfplot['posrtn'] = dfplot['value'] > 0
dfplot.dropna(inplace=True)

sb.boxplot(data=dfplot, x='group', y='posrtn', hue='cpiyoy1m.above')
