#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:29:25 2024

@author: lcota
"""

sector_meta = {
    'spx': 'SP500 Index',
    's5cond': 'Consumer Discretionary',
    's5cons': 'Consumer Staples',
    's5enrs': 'Energy',
    's5finl': "Financial",
    "s5hlth": "Healthcare",
    "s5indu": "Industrial",
    "s5matr": "Materials",
    "s5tech": "Technology"
    }

etf_meta = {
    'IVV': "SP500 Index",
    'IYW': "Technology",
    'SOXX': "Semi-conductors",
    'IGV': 'Expanded Tech-Software',
    'IHI': 'Medical Devices',
    'IGM': 'Expanded Tech Sector',
    'IYH': "US Health Care",
    'XT': 'Exponential Technologies',
    'IYF': 'Financials',
    'IYJ': 'Industrials',
    'IYG': 'Financial Services',
    'IYK': 'Consumer Staples',
    'IYC': 'Consumer Discretionary'
    }
# ETFS:
    # "IVV",  # SP500 broad market
    # "IYW",  # Technology
    # "SOXX",  # Semiconductors
    # "IGV",  # Expanded Tech-Software ETF
    # "IHI",  # Medical Devices
    # "IGM",  # Expanded Tech Sector
    # "IYH",  # US Health Care
    # "XT",  # Exponential Technologies (wtf is this)
    # "IYF",  # Financials
    # "IYJ",  # Industrials
    # "IYG",  # Financial Services
    # "IYK",  # Consumer Staples
    # "IYC",  # Consumer Discretionary

#%% Imports 
from pandas import DataFrame, read_parquet, read_csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
import datetime
import statsmodels.api as sm
import scipy
from scipy.stats import linregress
import numpy as np
import pandas as pd
import pickle
import shelve

from utils import *

pd.options.display.max_columns = 20
pd.options.display.width = 180

#%% Load Data
dfcpi = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/cpi.pq")
dfpmi = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/pmi.pq")
dfnmi = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/nmi.pq")
trdf = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/blk_etf_totrtn_pivot.pq")
cpifred = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/cpifred.pq")

nmi_full = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/ism_nmi_full.pq")
pmi_full = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/ism_pmi_full.pq")
trsectors = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/sector_returns.pq")



blktr = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/data/processed/blk_etf_totrtn.pq")

monthly_returns = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/monthly_returns.pq")
monthly_tr = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/monthly_tr.pq")

dftrcpi = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/dftrcpi.pq")
dfrtncpi = read_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/dfrtncpi.pq")


#%% Plot per symbol CPI regression line, CPI ~ returns
plt.ion()
tickers = ['IVV', 'IYW', 'SOXX', 'IGV', 'IGM', 'IYH', 'IYF', 'IYJ', 'IYG', 'IYK', 'IYC']

dfrtncpi.drop('date', axis=1, inplace=True)
dfrtncpi


sb.lmplot(dfrtncpi, x='cpiyoy', y='IVV')

lreg_results = []
for ticker in tickers:
    # sb.regplot(dfrtncpi, x='cpiyoy', y=ticker)
    # plt.figure()
    results = linregress(dfrtncpi['cpiyoy'], dfrtncpi[ticker])
    lreg_results.append(results)


dfresults = pd.DataFrame({'slope': [np.round(x.slope, 4) for x in lreg_results],
             'intercept': [np.round(x.intercept, 4) for x in lreg_results],
             'rvalue': [np.round(x.rvalue, 4) for x in lreg_results],
             'pvalue': [np.round(x.pvalue, 4) for x in lreg_results],
             'stderr': [np.round(x.stderr, 4) for x in lreg_results]})

dfresults['ticker'] = tickers
dfresults

#%% Summary table with mean sector return for high/med/low CPI

dfrtncpi.head()
mean_returns = dfrtncpi.drop('cpiyoy', axis=1).groupby('cpi.hml').agg('mean')[tickers]

#%% Barplot of returns x regime for each ETF
mean_returns_long = pd.melt(mean_returns.reset_index(), var_name='ticker', id_vars='cpi.hml', value_name='mean_rtn')
mean_returns_long['desc'] = mean_returns_long['ticker'].apply(lambda x: etf_meta[x])
 
plt.ioff()
sb.barplot(mean_returns_long, x='desc', y='mean_rtn', hue='cpi.hml')
plt.grid(ls='--', alpha=.5)
plt.xticks(rotation=45)
plt.show()


#%% Generate betas for standard, unconditional returns for all sectors relative to spx

betas = {}
models = {}
dfbetas, dfmodels = calculate_betas(dfreturns_wide, benchmark='spx', label='all')

betas['ALL'] = dfbetas 
models['ALL'] = dfmodels
dfbetas

#%% calculate betas for CPI, NMI, PMI regimes
feature_cols = ['cpi.hml', 'pmi.hml', 'nmi.hml']
values = ['LOW', 'MED', 'HIGH']

for feature_col in feature_cols:
    label_prefix = feature_col.split(".")[0].upper()
    for value in values:
        index = dfhml[dfhml[feature_col] == value].index
        label = f"{label_prefix.upper()}.{value.upper()}"
        dfbetas, dfmodels = calculate_betas(dfreturns_wide.loc[index], 
                                            benchmark='spx', 
                                            label=label)
        betas[label] = dfbetas
        models[label] = dfmodels


dfallbetas = pd.concat([betas[k] for k in betas.keys()])

with shelve.open("workspace/trial1/trial1.db", flag='c') as db:
    db["dfbetas"] = dfbetas
    db["beta_models"] = models
    
