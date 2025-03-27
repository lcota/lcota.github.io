#%% Imports
from importlib import reload

import datatable as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import cvxpy as cp
import scipy.sparse as sp
import statsmodels.api as sm

import datatools
from datatools import *
import strategy
from strategy import *




#%% load data 
dfx = pd.read_parquet("data/workspace/dfX.pq").reset_index()
dfy = pd.read_parquet("data/workspace/dfY.pq").reset_index()
dftr = pd.read_parquet("data/workspace/dftotrtn.pq").reset_index()
dfref = pd.read_csv("data/processed/etfref.csv")
dfref.set_index('ticker', inplace=True)
# dfx = dfx[['refmonthyear', 'cpiyoy', 'pmnmi', 'pmpmi', 'cpiyoy.surprise']]
dfx.set_index('refmonthyear', inplace=True)
dfy.set_index('refmonthyear', inplace=True)

# dtx = dt.Frame(dfx)
# dty = dt.Frame(dfy)
# dttr = dt.Frame(dftr)


#%% create signal
# pmi_thresh = 50
# cpi_thresh = dfx['cpiyoy'].median()
cpi_thresh = roll_quantile(dfx['cpiyoyact'], colname='cpiyoy.q50')
pmi_thresh = roll_quantile(dfx['pmiact'], colname='pmi.q50')
nmi_thresh = roll_quantile(dfx['nmiact'], colname='nmi.q50')

dfx['cpi.thresh'] = cpi_thresh
dfx['pmi.thresh'] = pmi_thresh
dfx['nmi.thresh'] = nmi_thresh


dfx['pmi.above'] = dfx['pmiact'] > dfx['pmi.thresh']
dfx['nmi.above'] = dfx['nmiact'] > dfx['nmi.thresh']
dfx['cpi.above'] = (dfx['cpiyoyact'] > dfx['cpi.thresh']).astype(int)

dfx['pmi1m.above'] = dfx['pmi1m'] > dfx['pmi.thresh']
dfx['pmi3m.above'] = dfx['pmi3m'] > dfx['pmi.thresh']
dfx['pmi6m.above'] = dfx['pmi6m'] > dfx['pmi.thresh']
dfx['pmi9m.above'] = dfx['pmi9m'] > dfx['pmi.thresh']
dfx['pmi12m.above'] = dfx['pmi12m'] > dfx['pmi.thresh']

dfx['nmi1m.above'] = dfx['nmi1m'] > dfx['nmi.thresh']
dfx['nmi3m.above'] = dfx['nmi3m'] > dfx['nmi.thresh']
dfx['nmi6m.above'] = dfx['nmi6m'] > dfx['nmi.thresh']
dfx['nmi9m.above'] = dfx['nmi9m'] > dfx['nmi.thresh']
dfx['nmi12m.above'] = dfx['nmi12m'] > dfx['nmi.thresh']

dfx['cpiyoy1m.above'] = dfx['cpiyoy1m'] > dfx['cpi.thresh']
dfx['cpiyoy3m.above'] = dfx['cpiyoy3m'] > dfx['cpi.thresh']
dfx['cpiyoy6m.above'] = dfx['cpiyoy6m'] > dfx['cpi.thresh']
dfx['cpiyoy9m.above'] = dfx['cpiyoy9m'] > dfx['cpi.thresh']
dfx['cpiyoy12m.above'] = dfx['cpiyoy12m'] > dfx['cpi.thresh']

# differential b/w consensus and forecasts
#
dfx['cpiyoy1m.diff'] = dfx['cpiyoy1m'] - dfx['cpiyoycons1m']
dfx['cpiyoy3m.diff'] = dfx['cpiyoy3m'] - dfx['cpiyoycons3m']
dfx['cpiyoy6m.diff'] = dfx['cpiyoy6m'] - dfx['cpiyoycons6m']
dfx['cpiyoy9m.diff'] = dfx['cpiyoy9m'] - dfx['cpiyoycons9m']
dfx['cpiyoy12m.diff'] = dfx['cpiyoy12m'] - dfx['cpiyoycons12m']

dfx['pmi1m.diff'] = dfx['pmi1m'] - dfx['pmicons1m']
dfx['pmi3m.diff'] = dfx['pmi3m'] - dfx['pmicons3m']
dfx['pmi6m.diff'] = dfx['pmi6m'] - dfx['pmicons6m']
dfx['pmi9m.diff'] = dfx['pmi9m'] - dfx['pmicons9m']
dfx['pmi12m.diff'] = dfx['pmi12m'] - dfx['pmicons12m']

dfx['nmi1m.diff'] = dfx['nmi1m'] - dfx['nmicons1m']
dfx['nmi3m.diff'] = dfx['nmi3m'] - dfx['nmicons3m']
dfx['nmi6m.diff'] = dfx['nmi6m'] - dfx['nmicons6m']
dfx['nmi9m.diff'] = dfx['nmi9m'] - dfx['nmicons9m']
dfx['nmi12m.diff'] = dfx['nmi12m'] - dfx['nmicons12m']

# dfx['pmi.above'] = (dfx['pmpmi'] > pmi_thresh).astype(int)
# dfx['nmi.above'] = (dfx['pmnmi'] > nmi_thresh).astype(int)
# dfx['cpi.above'] = (dfx['cpiyoy'] > cpi_thresh).astype(int)


# dfx.dropna(inplace=True)
# dfy.dropna(inplace=True)

# dfall = dfx['wgt.combo.signal'].copy()
# dfall['spy'] = dfy['spy']