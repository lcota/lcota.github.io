# %% module imports
import yfinance as yf
from yfinance.ticker import Ticker
import pandas as pd
from pandas import read_csv
from modin import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
sys.path.append("/Users/lcota/Dropbox/prj/Turnleaf_Analytics/")

from utils import *

import funcs

from funcs import *
from datatools import *

import datetime
import dateutil

# %%
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 180)

spindices = {
    "^GSPC": "S&P 500",
    "^GSPE": "Energy",
    "^SP500-20": "Industrials",
    "^SP500-60": "Real Estate",
    "^SP500-40": "Financials",
    "^SP500-15": "Materials",
    "^SP500-25": "Consumer Discretionary",
    "^SP500-45": "Information Technology",
    "^SP500-35": "Healthcare",
}


# SPDR ETFS
## Tickers for SP500 sector ETFs
spdr_tickers = [
    "XLC",  # Communication Services
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLE",  # Energy
    "XLF",  # Financials
    "XLV",  # Health Care
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLRE",  # Real Estate
    "XLU",  # Utilities⁄
    "SPY",  # broad market
]

# Vanguard ETFS
vgrd_tickers = [
    "VOO",  # SP500 broad market
    "VGT",  # Information Technology
    "VHT",  # Health Care
    "VFH",  # Financials
    "VDE",  # Energy
    "VDC",  # Cons Staples
    "VCR",  # Cons Discretionary
    "VPU",  # Utilities
    "VOX",  # Communication Services
    "VAW",  # Materials
]

# BlackRock iShares ETFs
blk_tickers = [
    "IVV",  # SP500 broad market
    "IYW",  # Technology
    "SOXX",  # Semiconductors
    "IGV",  # Expanded Tech-Software ETF
    "IHI",  # Medical Devices
    "IGM",  # Expanded Tech Sector
    "IYH",  # US Health Care
    "XT",  # Exponential Technologies (wtf is this)
    "IYF",  # Financials
    "IYJ",  # Industrials
    "IYG",  # Financial Services
    "IYK",  # Consumer Staples
    "IYC",  # Consumer Discretionary
]


# %%
# --------------------------------------------------------------------------
# Define data download functions


def get_data(ticker, start="2001-01-01"):
    df = yf.download(ticker, start)
    df["Ticker"] = ticker

    df.reset_index(inplace=True)
    df.columns = [c.lower().replace(" ", "") for c in df.columns]

    return df


def get_mult_tickers(tickers):
    data = []
    for t in tickers:
        data.append(get_data(t))

    df = pd.concat(data)
    return df


# %%
# --------------------------------------------------------------------------
# Download and store data
spdr_df = get_mult_tickers(spdr_tickers)
vgrd_df = get_mult_tickers(vgrd_tickers)
blk_df = get_mult_tickers(blk_tickers)


spdr_df.to_parquet("/Users/lcota/prj/Turnleaf_Analytics/data/raw/spdr_etfs.pq")
vgrd_df.to_parquet("/Users/lcota/prj/Turnleaf_Analytics/data/raw/vgrd_etfs.pq")
blk_df.to_parquet("/Users/lcota/prj/Turnleaf_Analytics/data/raw/blk_etfs.pq")


# %%
# --------------------------------------------------------------------------
# Create total return series & cache


def _calc_monthly_returns(ticker, df):
    rtn = df[df.ticker == ticker][["date", "ticker", "adjclose"]].copy()
    rtn["year"] = rtn["date"].apply(lambda x: x.year)
    rtn["month"] = rtn["date"].apply(lambda x: x.month)
    rtn.set_index("date", inplace=True)

    rtn = rtn.asfreq("BME").set_index(["year", "month"]).adjclose.pct_change()
    rtn["ticker"] = ticker
    return rtn


def calc_monthly_returns(df):
    tickers = df["ticker"].unique()
    monthly_returns = [_calc_monthly_returns(ticker, df) for ticker in tickers]
    monthly_returns = pd.concat(monthly_returns)

    return monthly_returns


def calc_tr(ticker, df):
    """Calculates total return based on adjusted close of ticker"""
    tr = df[df.ticker == ticker][["date", "ticker", "adjclose"]].copy()
    tr["pctchg"] = tr["adjclose"].pct_change()
    _tr = tr["pctchg"].values
    _tr[0] = 0
    _tr = 100 * np.cumprod(1 + _tr)
    tr["totrtn"] = _tr
    # tr['totrtn'] = tr['adjclose'].pct_change().apply(lambda x: 100*np.cumprod(1+x))
    # tr.drop('adjclose', axis=1, inplace=True)
    # tr = tr.dropna()

    return tr


def calc_tot_returns(df):
    tickers = df["ticker"].unique()
    totrtns = [calc_tr(ticker, df) for ticker in tickers]
    totrtns = pd.concat(totrtns)

    return totrtns


totrtns = calc_tot_returns(blk_df)
totrtns.to_parquet(
    "/Users/lcota/prj/Turnleaf_Analytics/data/processed/blk_etf_totrtn.pq"
)

trdf = totrtns.pivot(index="date", columns="ticker", values="totrtn")
trdf.to_parquet(
    "/Users/lcota/prj/Turnleaf_Analytics/data/processed/blk_etf_totrtn_pivot.pq"
)

# trdf = pd.read_parquet("/Users/lcota/prj/Turnleaf_Analytics/data/processed/blk_etf_totrtn_pivot.pq")
# trdf.reset_index(inplace=True)
# trdf['refmonthyear'] = trdf['date'].apply(lambda x: x.strftime("%b.%Y").upper())
# trdf.set_index('date', inplace=True)

# %% Create monthly returns incorporating prior month closing values
# %% Drop XT and IHI b/c of short history
trdf.drop("XT", inplace=True, axis=1)
trdf.drop("IHI", inplace=True, axis=1)
trdf = trdf.dropna()


# %% create monthly returns series
trdf.set_index("date", inplace=True)
trdf.resample("BME", convention="start").first()
trdf.resample("BME", convention="start").last()

trdf.reset_index(inplace=True)
trdf["year"] = trdf["date"].apply(lambda x: x.year)
trdf["month"] = trdf["date"].apply(lambda x: x.month)
trdf["day"] = trdf["date"].apply(lambda x: x.day)

# SOXX FIRST 2024-04-30    1635.584272
# SOXX LAST 2024-04-30    1421.303366

# trdf[(trdf.year == 2024) & (trdf.month==4)][['date', 'SOXX']]

"""
ticker              IGM          IGV  ...         IYW         SOXX
date                                  ...                         
2001-07-31   103.505198    97.353892  ...   83.211713   101.720551
2001-08-31    89.222123    77.606647  ...   72.262805    94.755702
2001-09-28    70.946563    59.241708  ...   56.423390    66.001389
2001-10-31    82.755402    73.143754  ...   68.613161    80.137656
2001-11-30    96.063741    85.031626  ...   78.832175    93.379212
                ...          ...  ...         ...          ...
2023-12-29  1630.814586   972.519884  ...  829.713260  1374.358997
2024-01-31  1703.443790  1007.858938  ...  854.047019  1398.095947
2024-02-29  1839.132787  1031.570218  ...  900.956992  1555.642505
2024-03-29  1882.351030  1022.171987  ...  913.452331  1618.533468
2024-04-30  1747.803696   941.376428  ...  843.722644  1421.303366
"""

trdf.set_index("date", inplace=True)
monthly_closes = trdf.resample("BME").last()
monthly_closes.drop(["year", "month", "day"], axis=1, inplace=True)
monthly_returns = monthly_closes.pct_change().dropna()
monthly_returns.reset_index(inplace=True)
monthly_returns["refmonthyear"] = monthly_returns["date"].apply(
    lambda x: x.strftime("%b.%Y").upper()
)


monthly_tr = monthly_returns.apply(lambda x: (1 + x).cumprod()) * 100
monthly_tr.reset_index(inplace=True)
monthly_tr["refmonthyear"] = monthly_tr["date"].apply(
    lambda x: x.strftime("%b.%Y").upper()
)

monthly_returns.to_parquet(
    "/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/monthly_returns.pq"
)
monthly_tr.to_parquet(
    "/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/monthly_tr.pq"
)
# %%
# --------------------------------------------------------------------------
# Load CPI/ISM data
dfcpi = pd.read_parquet("/Users/lcota/prj/Turnleaf_Analytics/data/processed/cpi_all.pq")
dfcpi = filter_cols(dfcpi, "usa")
dfcpi.reset_index(inplace=True)
dfcpi["pubdate"] = dfcpi["usa.cpi.yoy.nsa.pubdate"].apply(adj_date)
dfcpi["refdate"] = dfcpi["usa.cpi.yoy.nsa.refdate"].copy()
dfcpi["index"] = dfcpi["pubdate"].copy()
# dfcpi['adjpubdate'] = dfcpi['usa.cpi.yoy.nsa.pubdate'].apply(funcs.adj_date)
dfcpi = dfcpi.set_index("index")


dfism_man = pd.read_parquet("/Users/lcota/prj/Turnleaf_Analytics/data/processed/ism_man.pq")
dfism_man = funcs.filter_cols(dfism_man, "usa")
dfism_man.reset_index(inplace=True)
dfism_man["pubdate"] = dfism_man["usa.ism.man.pubdate"].apply(funcs.adj_date)
dfism_man["refdate"] = dfism_man["usa.ism.man.refdate"].copy()
dfism_man["index"] = dfism_man["pubdate"].copy()
dfism_man = dfism_man.set_index("index")

dfism_nonman = pd.read_parquet("/Users/lcota/prj/Turnleaf_Analytics/data/processed/ism_nonman.pq")
dfism_nonman.reset_index(inplace=True)
dfism_nonman["pubdate"] = dfism_nonman["pubdate"].apply(funcs.adj_date)
dfism_nonman["refdate"] = dfism_nonman["usa.ism.nonman.refdate"].copy()
dfism_nonman["index"] = dfism_nonman["pubdate"].copy()
dfism_nonman = dfism_nonman.set_index("index")
dfism_nonman = funcs.filter_cols(dfism_nonman, "usa")

cpifred = read_csv("/Users/lcota/prj/Turnleaf_Analytics/data/raw/cpi_fred.csv")
cpifred["date"] = cpifred["date"].apply(pd.to_datetime)
cpifred["refmonthyear"] = cpifred["date"].apply(lambda x: x.strftime("%b.%Y").upper())
cpifred["year"] = cpifred["date"].apply(lambda x: x.year)
cpifred["cpiyoy"] = cpifred["usa.cpi"].pct_change(periods=12) * 100
cpifred.set_index("date", inplace=True)

dfcpi["refmonthyear"] = dfcpi["refdate"].apply(lambda x: x.strftime("%b.%Y").upper())
dfism_man["refmonthyear"] = dfism_man["refdate"].apply(
    lambda x: x.strftime("%b.%Y").upper()
)
dfism_nonman["refmonthyear"] = dfism_nonman["refdate"].apply(
    lambda x: x.strftime("%b.%Y").upper()
)

#%% store cpi data
dfcpi.to_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/cpi.pq")
dfism_man.to_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/pmi.pq")
dfism_nonman.to_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/nmi.pq")
cpifred.to_parquet("/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/cpifred.pq")

#
# %% Merge monthly returns with CPI data
# --------------------------------------------------------------------------
monthly_returns.set_index("refmonthyear", inplace=True)
monthly_tr.set_index("refmonthyear", inplace=True)


cpifred.reset_index(inplace=True)
cpifred.set_index("refmonthyear", inplace=True)
cpifred

dfcpi.reset_index(inplace=True)
dfcpi.set_index("refmonthyear", inplace=True)
dfcpi

dftrcpi = pd.merge(
    monthly_tr, cpifred["cpiyoy"], left_index=True, right_index=True, how="inner"
)

dfrtncpi = pd.merge(
    monthly_returns, cpifred["cpiyoy"], left_index=True, right_index=True, how="inner"
)


dfrtncpi["cpi.hml"] = dfrtncpi["cpiyoy"].apply(hml)
dfrtncpi["cpi.hml"] = pd.Categorical(
    dfrtncpi["cpi.hml"], categories=["LOW", "MED", "HIGH"], ordered=True
)
dftrcpi["cpi.hml"] = dftrcpi["cpiyoy"].apply(hml)
dftrcpi["cpi.hml"] = pd.Categorical(
    dftrcpi["cpi.hml"], categories=["LOW", "MED", "HIGH"], ordered=True
)

dftrcpi.to_parquet(
    "/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/dftrcpi.pq"
)
dfrtncpi.to_parquet(
    "/Users/lcota/prj/Turnleaf_Analytics/sector_etfs/workspace/dfrtncpi.pq"
)


# add HI MED LOW to cpifred data
cpifred.reset_index(inplace=True)
cpifred["refmonthyear"] = cpifred["date"].apply(lambda x: x.strftime("%b.%Y").upper())
cpifred.set_index("date", inplace=True)
cpifred = cpifred[cpifred.cpiyoy.isna() == False]
cpifred["cpi.hml"] = cpifred["cpiyoy"].apply(hml)
cpifred.to_parquet("sector_etfs/workspace/cpifred.pq")


# %% Generate monthly returns
trsectors = pd.read_parquet("workspace/sector_returns.pq")
mindate = (
    trsectors.groupby("ticker")["date"].agg(["min", "max"]).reset_index()["min"].max()
)
maxdate = (
    trsectors.groupby("ticker")["date"].agg(["min", "max"]).reset_index()["max"].min()
)
trsectors = trsectors[(trsectors.date >= mindate) & (trsectors.date <= maxdate)]

daily_sector_levels = trsectors.copy()
daily_sector_levels.to_parquet("workspace/daily_sector_levels.pq")


# calculate monthly returns
closes = (
    (
        trsectors.pivot(
            columns="ticker", values="tot_rtn_gross_dvd", index=["date", "refmonthyear"]
        )
        .reset_index()
        .set_index("date")
        .resample("BME")
    )
    .last()
    .reset_index()
    .set_index(["date", "refmonthyear"])
)

monthly_returns = closes.pct_change().dropna()
monthly_returns.to_parquet("workspace/monthly_returns.pq")

# pivot daily levels to wide format by ticker
dfdaily = pd.read_parquet("workspace/daily_sector_levels.pq")
dfdaily = dfdaily.pivot(index=["date", "refmonthyear"], columns="ticker", values="tot_rtn_gross_dvd")
dfdaily = dfdaily.dropna()
dfdaily.to_parquet("workspace/daily_levels_wide.pq")
daily_returns = dfdaily.pct_change().dropna()
daily_returns.to_parquet("workspace/daily_returns.pq")

# Calculate forward returns
# will include days from 1-20 for intra-month studies
# will include 1,2,3,6,9,12 months forward returns.


def calc_forward_returns(data, ticker, daily_periods = 20, monthly_periods = 12):
    df = data[data['ticker'] == ticker].copy()
    df.set_index('date', inplace=True)

    for i in range(1, 1 + daily_periods):
        colname = f"rtn_{i}D"
        df[colname] = (df['level'].shift(-i)/df['level']) - 1

    for i in range(1, 1 + monthly_periods):
        # assumes months are all 21 days long
        colname = f"rtn_{i}M"
        df[colname] = (df['level'].shift(-i)/df['level']) - 1

    return df


dfdaily = pd.read_parquet("workspace/daily_sector_levels.pq").drop('refmonthyear', axis=1)
dfdaily.columns = ['date', 'level', 'ticker']


head(dfdaily)
tickers = dfdaily['ticker'].unique()

fwd_returns = {}
for ticker in tickers:
    fwd_returns[ticker] = calc_forward_returns(data=dfdaily, ticker=ticker)
 
df_fwdreturns = pd.concat([fwd_returns[k] for k in fwd_returns.keys()])

head(df_fwdreturns)
df_fwdreturns['px_date'] = df_fwdreturns.index.copy() 
df_fwdreturns['px_date'] = df_fwdreturns['px_date'].apply(lambda x: pd.to_datetime(x).date())
df_fwdreturns['refmonthyear'] = df_fwdreturns['px_date'].apply(lambda x: x.strftime("%b.%Y").upper())
df_fwdreturns.to_parquet("workspace/fwd_returns.pq")




#
#
#
# ----------------------------------------------------
# new section
