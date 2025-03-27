#%% imports 
import datetime
from importlib import reload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


import datatools
from datatools import *
import strategy
from strategy import *

#%% transform cpi data for workspace
dfcpi = pd.read_parquet("data/processed/cpi_all.pq")
# dfcpi = funcs.filter_cols(dfcpi, "usa")
# dfcpi.reset_index(inplace=True)
# dfcpi['pubdate'] = dfcpi['usa.cpi.yoy.nsa.pubdate'].apply(funcs.adj_date)
# dfcpi['index'] = dfcpi['pubdate'].copy()
# # dfcpi['adjpubdate'] = dfcpi['usa.cpi.yoy.nsa.pubdate'].apply(funcs.adj_date)
# dfcpi = dfcpi.set_index('index')