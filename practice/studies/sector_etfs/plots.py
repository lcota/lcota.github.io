# -*- coding: utf-8 -*-
# Data sets used: dfcpi, dfism_man, dfism_nonman, trdf
# Data for ISM, CPI begins in 2013
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
from pandas import DataFrame, read_parquet, read_csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.font_manager as fm
fm.fontManager.addfont("/Users/lcota/Library/Fonts/Raleway-Regular.ttf")
fm.fontManager.addfont("/Users/lcota/Library/Fonts/Raleway-Light.ttf")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Raleway'
plt.rcParams['font.style'] = "normal"


# from data import hml
from utils import *

#%% Load data sets
dfcpi = read_parquet("workspace/cpi.pq")
dfpmi = read_parquet("workspace/pmi.pq")
dfnmi = read_parquet("workspace/nmi.pq")
trdf = read_parquet("workspace/blk_etf_totrtn_pivot.pq")
cpifred = read_parquet("workspace/cpifred.pq")

dfism = read_parquet("workspace/ism_df.pq")
nmi_full = read_parquet("workspace/ism_nmi_full.pq")
pmi_full = read_parquet("workspace/ism_pmi_full.pq")
trsectors = read_parquet("workspace/sector_returns.pq")

monthly_returns = pd.read_parquet("workspace/monthly_returns.pq")
cpifred = pd.read_parquet("workspace/cpifred.pq")


#%% Examine returns 

dfcpi.head()
cpiyoy = dfcpi[['pubdate', 'refdate', 'refmonthyear', 'usa.cpi.yoy.nsa.realised']].copy()
cpiyoy
cpiyoy['usa.cpi.yoy.nsa.realised'].plot()


#%% Shaded plot / high, medium, low inflation

plt.ioff()

width_px = 410
height_px = 171
dpi = 100

width_in = 5.6
height_in = 2.5

plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titlecolor'] = '#595959'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelcolor'] = '#595959'
plt.rcParams['xtick.color'] = '#595959'
plt.rcParams['ytick.color'] = '#595959'


figsize = (width_in, height_in)

fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
ax.plot(cpiyoy.index, cpiyoy['usa.cpi.yoy.nsa.realised'].values, color='black')
ax.axhline(1, color='black', alpha=.5, ls='--', lw=1)
ax.axhline(3, color='black', alpha=.5, ls='--', lw=1)

ax.fill_between(x=cpiyoy.index, y1=3, y2=10, facecolor='red', alpha=.2)
ax.fill_between(x=cpiyoy.index, y1=1, y2=3, facecolor='yellow', alpha=.2)
ax.fill_between(x=cpiyoy.index, y1=-1, y2=1, facecolor='green', alpha=.2)

plt.title("Inflation Regimes - US CPI YoY")
plt.show()

#%% Plot of CPI using longer history (FRED CPI data)

dpi = 240

width_in = 5.6
height_in = 2.5

plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.titlecolor'] = '#595959'
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.labelcolor'] = '#595959'
plt.rcParams['xtick.color'] = '#595959'
plt.rcParams['ytick.color'] = '#595959'

plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

figsize = (width_in, height_in)

fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

mask = cpifred.index >= pd.Timestamp(year=1989, month=1, day=1)

miny = cpifred[mask]['cpiyoy'].min()
maxy = cpifred[mask]['cpiyoy'].max()
ax.plot(cpifred[mask].index, cpifred[mask]['cpiyoy'].values, color='black', lw=1, alpha=.7)
ax.axhline(1, color='black', alpha=.5, ls='--', lw=1)
ax.axhline(3, color='black', alpha=.5, ls='--', lw=1)

ax.fill_between(x=cpifred[mask].index, y1=3, y2=maxy, facecolor='red', alpha=.2)
ax.fill_between(x=cpifred[mask].index, y1=1, y2=3, facecolor='yellow', alpha=.2)
ax.fill_between(x=cpifred[mask].index, y1=miny, y2=1, facecolor='green', alpha=.2)

plt.title("Inflation Regimes - US CPI YoY")

plt.show()

#%% Create HML regime for CPI & monthly returns
dfhml = pd.merge(monthly_returns.reset_index(), 
                  cpifred[['refmonthyear', 'cpi.hml']], 
                  left_on='refmonthyear',
                  right_on='refmonthyear',
                  how='left').set_index(['date', 'refmonthyear']).dropna()

dfhml['cpi.hml'] = pd.Categorical(dfhml['cpi.hml'], 
                                  categories=['LOW', 'MED', 'HIGH'],
                                  ordered=True)


long_returns = pd.melt(dfhml, 
                       id_vars='cpi.hml', 
                       var_name='ticker', 
                       value_name='rtn', 
                       ignore_index=False).dropna()


mean_returns = long_returns.reset_index().groupby(['ticker', 'cpi.hml'], observed=True)['rtn'].agg('mean').reset_index()


#%% Barplot of mean sector return x inflationary regime
fig, ax = plt.subplots() 
plt.ioff()
sb.barplot()
sb.barplot(mean_returns, x='ticker', y='rtn', hue='cpi.hml')
plt.grid(ls='--', alpha=.5)
plt.xticks(rotation=45)
plt.title("Mean Monthly Return x Inflation Regime")
plt.show()


#%% shaded hi/med/low plots of ISM PMI
# plt.ion()
lq = 1/4
uq = 3/4

lb, ub = pmi_full['pmi'].quantile([lq, uq])
pmi_full['pmi.hml'] = pmi_full['pmi'].apply(lambda x: hml(x, lb=lb, ub=ub))
pmi_full['pmi.hml'] = pd.Categorical(pmi_full['pmi.hml'], categories=['LOW', 'MED', 'HIGH'], ordered=True)

plt.ioff()
fig, ax = plt.subplots()
miny = pmi_full['pmi'].min()
maxy = pmi_full['pmi'].max()
ax.plot(pmi_full['date'], pmi_full['pmi'].values, color='black')
ax.axhline(lb, color='black', alpha=.5, ls='--', lw=1)
ax.axhline(ub, color='black', alpha=.5, ls='--', lw=1)

ax.fill_between(x=pmi_full['date'], y1=ub, y2=maxy, facecolor='green', alpha=.2)
ax.fill_between(x=pmi_full['date'], y1=lb, y2=ub, facecolor='yellow', alpha=.2)
ax.fill_between(x=pmi_full['date'], y1=miny, y2=lb, facecolor='red', alpha=.2)

plt.title("PMI Regimes - US PMI")

plt.show()

#%% shaded hi/med/low plot of ISM NMI
lq = 1/4
uq = 3/4

lb, ub = nmi_full['nmi'].quantile([lq, uq])
nmi_full['nmi.hml'] = nmi_full['nmi'].apply(lambda x: hml(x, lb=lb, ub=ub))
nmi_full['nmi.hml'] = pd.Categorical(nmi_full['nmi.hml'], categories=['LOW', 'MED', 'HIGH'], ordered=True)


plt.ioff()
fig, ax = plt.subplots()
miny = nmi_full['nmi'].min()
maxy = nmi_full['nmi'].max()
ax.plot(nmi_full['date'], nmi_full['nmi'].values, color='black')
ax.axhline(lb, color='black', alpha=.5, ls='--', lw=1)
ax.axhline(ub, color='black', alpha=.5, ls='--', lw=1)

ax.fill_between(x=nmi_full['date'], y1=ub, y2=maxy, facecolor='green', alpha=.2)
ax.fill_between(x=nmi_full['date'], y1=lb, y2=ub, facecolor='yellow', alpha=.2)
ax.fill_between(x=nmi_full['date'], y1=miny, y2=lb, facecolor='red', alpha=.2)

plt.title("NMI Regimes - US NMI")

plt.show()

#%% Create df with HML for CPI, PMI, NMI

# add NMI & PMI HML values
dfhml.reset_index(inplace=True)
monthly_returns.set_index('refmonthyear', inplace=True)


dfhml.set_index('refmonthyear', inplace=True)
nmi_full.set_index('refmonthyear', inplace=True)
pmi_full.set_index('refmonthyear', inplace=True)

dfhml['nmi.hml'] = nmi_full['nmi.hml']
dfhml['pmi.hml'] = pmi_full['pmi.hml']
dfhml = dfhml.reset_index().set_index(['date', 'refmonthyear'])


long_returns = pd.melt(dfhml, 
                       id_vars=['nmi.hml', 'pmi.hml', 'cpi.hml'], 
                       var_name='ticker', 
                       value_name='rtn', 
                       ignore_index=False)

#%% plots of returns grouped by PMI regime
dfplot = (long_returns[long_returns['pmi.hml'].isna() == False]
          .groupby(['ticker', 'pmi.hml'], observed=False)['rtn']
          .agg('mean').reset_index()).dropna()

fig, ax = plt.subplots() 
plt.ioff()
sb.barplot()
sb.barplot(dfplot, x='ticker', y='rtn', hue='pmi.hml')
plt.grid(ls='--', alpha=.5)
plt.xticks(rotation=45)
plt.title("Mean Monthly Return x PMI Regime")
plt.show()



#%% plots of returns grouped by NMI regime
dfplot = (long_returns[long_returns['nmi.hml'].isna() == False]
          .groupby(['ticker', 'nmi.hml'], observed=False)['rtn']
          .agg('mean').reset_index()).dropna()

fig, ax = plt.subplots() 
plt.ioff()
sb.barplot()
sb.barplot(dfplot, x='ticker', y='rtn', hue='nmi.hml')
plt.grid(ls='--', alpha=.5)
plt.xticks(rotation=45)
plt.title("Mean Monthly Return x NMI Regime")
plt.show()

#%% Save down dataframe with NMI/PMI/CPI Actual & monthly returns
long_returns.to_parquet("workspace/returns_with_hml.pq")
