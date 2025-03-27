# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 12:04:35 2023

@author: lcota
"""
import datetime
from importlib import reload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import yfinance as yf
from fredapi import Fred

import datatools
from datatools import *
import strategy
from strategy import *

fred = Fred(api_key="b3ce3f2171d3028e21081bc05946f7fc")

#%% read data
dfetfs = pd.read_parquet("data/processed/etfs.pq")
idx = pd.DatetimeIndex(dfetfs.date.values)
dfetfs = dfetfs.set_index(idx)

mindate = datetime.date(2014, 1, 1) # minimum data date for analysis
mints = pd.Timestamp(pd.to_datetime(mindate)) # minimum timestamp for analysis 

etfs2keep = ['SPY', 'AGG',  # sp500, Agg bond etf
            'XOP', 'XME', 'XHB', 'GLD',  # Oil & Gas, Metals & Mining, Homebuilders, Gold Shares
            'BIL', 'TIP', 'SPTS', 'SPTI'] # 1-3Mth TBills, TIPs, Short Term Tsy, Med. Term Tsy

dfcloses = dfetfs[dfetfs['date'].apply(pd.to_datetime) > mints]
dfcloses = dfcloses.pivot(columns='ticker', values='adjusted')[etfs2keep].dropna()

dftotrtn = dfcloses.pct_change().dropna().apply(lambda x: 100*np.cumprod(1 + x))

# Monthly volatilities (STDevs) -- maybe this is the forecast target? 
stdev_30d = (dftotrtn.pct_change()
             .rolling(window=30, min_periods=30)
             .std()
             .dropna()
             .apply(lambda x: np.round(x * 100 * np.sqrt(252), 2)))

del dfetfs

#%% Read in Turnleaf Data
cpi = pd.read_parquet("data/processed/cpi_all.pq")
cols2keep = [c for c in cpi.columns if c.startswith("usa")]
cpi = cpi[cols2keep]

pmi = pd.read_parquet("data/processed/ism_man.pq")
nmi = pd.read_parquet("data/processed/ism_nonman.pq")

cpi = cpi[[c for c in cpi.columns if 'nsa.date' not in c]]
cpi = cpi[[c for c in cpi.columns if ('cpi.mom' not in c and 'nsa.realised' not in c)]]
pmi = pmi[[c for c in pmi.columns if 'man.date' not in c]]
nmi = nmi[[c for c in nmi.columns if 'nonman.date' not in c]]

head(pmi)


cpi['refdate'] = cpi.index.values
cpi['refdate'] = cpi['refdate'].apply(lambda x: pd.to_datetime(x))
cpi['refmonthyear'] = cpi['refdate'].apply(lambda x :x.strftime("%b.%Y").upper())

pmi['refdate'] = pmi.index.values
pmi['refdate'] = pmi['refdate'].apply(lambda x: pd.to_datetime(x))
pmi['refmonthyear'] = pmi['refdate'].apply(lambda x :x.strftime("%b.%Y").upper())

nmi['refdate'] = nmi.index.values
nmi['refdate'] = nmi['refdate'].apply(lambda x: pd.to_datetime(x))
nmi['refmonthyear'] = nmi['refdate'].apply(lambda x :x.strftime("%b.%Y").upper())

cpi.reset_index(inplace=True)
nmi.reset_index(inplace=True)
pmi.reset_index(inplace=True)

cpi.to_parquet("data/workspace/cpi.pq")
nmi.to_parquet("data/workspace/nmi.pq")
pmi.to_parquet("data/workspace/pmi.pq")




#%% get yfinance and FRED data
spx = yf.Ticker('^GSPC').history(start=datetime.date(1947, 1, 1))

cpi = pd.read_csv("data/raw/cpi_fred.csv")
cpi['date'] = cpi['date'].apply(lambda x: pd.to_datetime(x).date())



df = spx.reset_index()[['Date', 'Close']].copy()
df.columns = ['date', 'spx.close']
df['date'] = df['date'].apply(lambda x: pd.to_datetime(x).date())


df = pd.merge(left=cpi, right=df, left_on='date', right_on='date', suffixes=[None, None])
df.set_index('date', inplace=True)
#%% plot surprises vs other variables
cpi.set_index('refmonthyear', inplace=True)
pmi.set_index("refmonthyear", inplace=True)
nmi.set_index("refmonthyear", inplace=True)

cols2keep = []
newnames = []   
for c in cpi.columns:
    tail = c.split(".")[-1]
    if tail[-1] == 'm':
        cols2keep.append(c)
        if 'cons' in c:
            newname = f"cpiyoy.cons.{tail}"
        else:
            newname = f"cpiyoy.pred.{tail}"
        newnames.append(newname)
    
cpi = cpi[cols2keep]
cpi.columns = newnames

head(cpi)


#%% inspect VIX and CPI/PMI data plots
vix = yf.Ticker('^VIX').history(start=datetime.date(2011, 1, 1))
vix.reset_index(inplace=True)

head(vix)
vix['Date']
vix['refmonthyear'] = vix_history['Date'].apply(lambda x: x.strftime("%b.%Y").upper())
vix_mthly = vix['Close'].resample(rule='1M').apply('ohlc')

head(vix_mthly)
vix_mthly['abschg'] = vix_mthly['close'] - vix_mthly['open']
vix_mthly['pctchg'] = vix_mthly['close'] / vix_mthly['open'] - 1
vix_mthly.reset_index(inplace=True)

vix_mthly['refmonthyear'] = vix_mthly['Date'].apply(lambda x: x.strftime("%b.%Y").upper())
vix_mthly.set_index('refmonthyear', inplace=True)

head(vix_mthly)
dfx['vix.abschg'] = vix_mthly['abschg']
dfx['vix.pctchg'] = vix_mthly['pctchg']

#%% plot vix vs other features

plt.title("VIX vs CPI / PMI Levels")
plt.subplot(321)
p = sb.scatterplot(dfx, x='cpiyoy', y='vix.abschg'); 
# plt.title("CPI YoY vs VIX abs Chg")
p.axhline(y=0, ls='--', color='r', alpha=.5)
p.axvline(x=2, ls='--', color='r', alpha=.5)

plt.subplot(322)
p = sb.scatterplot(dfx, x='cpiyoy', y='vix.pctchg'); 
# plt.title("CPI YoY vs VIX Pct Chg")
p.axhline(y=0, ls='--', color='r', alpha=.5)
p.axvline(x=2, ls='--', color='r', alpha=.5)

plt.subplot(323)
p = sb.scatterplot(dfx, x='pmnmi', y='vix.abschg'); 
# plt.title("PMNMI vs VIX abs Chg")
p.axhline(y=0, ls='--', color='r', alpha=.5)
p.axvline(x=50, ls='--', color='r', alpha=.5)

plt.subplot(324)
p = sb.scatterplot(dfx, x='pmnmi', y='vix.pctchg'); 
# plt.title("PMNMI vs VIX Pct Chg")
p.axhline(y=0, ls='--', color='r', alpha=.5)
p.axvline(x=50, ls='--', color='r', alpha=.5)

plt.subplot(325)
p = sb.scatterplot(dfx, x='pmpmi', y='vix.abschg'); 
# plt.title("PMPMI vs VIX abs Chg")
p.axhline(y=0, ls='--', color='r', alpha=.5)
p.axvline(x=50, ls='--', color='r', alpha=.5)

plt.subplot(326)
p = sb.scatterplot(dfx, x='pmpmi', y='vix.pctchg'); 
# plt.title("PMPMI vs VIX Pct Chg")
p.axhline(y=0, ls='--', color='r', alpha=.5)
p.axvline(x=50, ls='--', color='r', alpha=.5)




#%%
plt.subplot(121)
plt.scatter(dfx['cpiyoy'], dfx['vix.abschg'])
plt.title("CPI vs VIX Abs Chg")
plt.axhline(y=0, ls='--', color='r', alpha=.5)
plt.axvline(x=2, ls='--', color='r', alpha=.5)


plt.subplot(122)
plt.scatter(dfx['cpiyoy'], dfx['vix.pctchg'])
plt.title("CPI vs VIX Pct Chg")
plt.axhline(y=0, ls='--', color='r', alpha=.5)
plt.axvline(x=2, ls='--', color='r', alpha=.5)












