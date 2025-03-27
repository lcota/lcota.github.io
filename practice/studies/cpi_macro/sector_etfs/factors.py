from matplotlib import pyplot as plt
from matplotlib.pyplot import plot, scatter, show, ion, ioff
from modin import pandas as pd
import ray
ray.init()

from utils import *

cpi = pd.read_parquet("workspace/cpi_fred_factors.pq")
cpitl = pd.read_parquet("workspace/cpi.pq")
dfreturns = pd.read_parquet("workspace/returns_with_hml.pq")
dfdaily = pd.read_parquet("workspace/daily_levels_wide.pq")
fwd_rtn = pd.read_parquet("workspace/fwd_returns.pq")


cpi['usa.cpi'].plot()

# Add CPI Rising for 3mth/6mth/1y
cpi['3M_rising'] = cpi['usa.cpi'] > cpi['usa.cpi'].shift(3)
cpi['6M_rising'] = cpi['usa.cpi'] > cpi['usa.cpi'].shift(6)
cpi['9M_rising'] = cpi['usa.cpi'] > cpi['usa.cpi'].shift(9)
cpi['12M_rising'] = cpi['usa.cpi'] > cpi['usa.cpi'].shift(12)

cpi['3M_rising_yoy'] = cpi['cpiyoy'] > cpi['cpiyoy'].shift(3)
cpi['6M_rising_yoy'] = cpi['cpiyoy'] > cpi['cpiyoy'].shift(6)
cpi['9M_rising_yoy'] = cpi['cpiyoy'] > cpi['cpiyoy'].shift(9)
cpi['12M_rising_yoy'] = cpi['cpiyoy'] > cpi['cpiyoy'].shift(12)

cpi['3M_rising'] = pd.Categorical(cpi['3M_rising'], categories=[True, False], ordered=True)
cpi['6M_rising'] = pd.Categorical(cpi['6M_rising'], categories=[True, False], ordered=True)
cpi['9M_rising'] = pd.Categorical(cpi['9M_rising'], categories=[True, False], ordered=True)
cpi['12M_rising'] = pd.Categorical(cpi['12M_rising'], categories=[True, False], ordered=True)

cpi['3M_rising_yoy'] = pd.Categorical(cpi['3M_rising_yoy'], categories=[True, False], ordered=True)
cpi['6M_rising_yoy'] = pd.Categorical(cpi['6M_rising_yoy'], categories=[True, False], ordered=True)
cpi['9M_rising_yoy'] = pd.Categorical(cpi['9M_rising_yoy'], categories=[True, False], ordered=True)
cpi['12M_rising_yoy'] = pd.Categorical(cpi['12M_rising_yoy'], categories=[True, False], ordered=True)

cpi['3Mchg'] = cpi['usa.cpi'] - cpi['usa.cpi'].shift(3)
cpi['6Mchg'] = cpi['usa.cpi'] - cpi['usa.cpi'].shift(6)
cpi['9Mchg'] = cpi['usa.cpi'] - cpi['usa.cpi'].shift(9)
cpi['12Mchg'] = cpi['usa.cpi'] - cpi['usa.cpi'].shift(12)

cpi['3Mchg_yoy'] = cpi['cpiyoy'] - cpi['cpiyoy'].shift(3)
cpi['6Mchg_yoy'] = cpi['cpiyoy'] - cpi['cpiyoy'].shift(6)
cpi['9Mchg_yoy'] = cpi['cpiyoy'] - cpi['cpiyoy'].shift(9)
cpi['12Mchg_yoy'] = cpi['cpiyoy'] - cpi['cpiyoy'].shift(12)

cpi.to_parquet("workspace/cpi_fred_factors.pq")

# Create HML dataframe, dropping return columns
dfhml = dfreturns[["nmi.hml", "pmi.hml", "cpi.hml"]].copy().reset_index().dropna()
dfhml = dfhml.groupby(["date", "refmonthyear"]).first()

# %% use Turnleaf data to construct factors for rising_yoy over 3M/6M/9M/12M horizons for US CPI

cpitl = pd.read_parquet("workspace/cpi.pq")
cols2keep = {'pubdate': 'pubdate',
             'refdate': 'refdate', 
             'refmonthyear': 'refmonthyear', 
             'usa.cpi.yoy.nsa.pubdate': 'cpi.pubdate', 
             'usa.cpi.yoy.nsa.refdate': 'cpi.refdate', 
             'usa.cpi.yoy.nsa.realised': 'cpi.realised', 
             'usa.tlbt.cpi.yoy.nsa.1m': 'cpi.1mf.yoy',
             'usa.tlbt.cpi.yoy.nsa.3m': 'cpi.3mf.yoy', 
             'usa.tlbt.cpi.yoy.nsa.6m': 'cpi.6mf.yoy', 
             'usa.tlbt.cpi.yoy.nsa.9m': 'cpi.9mf.yoy', 
             'usa.tlbt.cpi.yoy.nsa.12m': 'cpi.12mf.yoy',
              # 'usa.tlbt.cons.cpi.yoy.nsa.1m': ,
              # 'usa.tlbt.cons.cpi.yoy.nsa.3m', 
              # 'usa.tlbt.cons.cpi.yoy.nsa.6m', 
              # 'usa.tlbt.cons.cpi.yoy.nsa.9m', 
              # 'usa.tlbt.cons.cpi.yoy.nsa.12m',
             }
cpitl = cpitl[cols2keep.keys()]
cpitl.head()
cpitl.rename(mapper=cols2keep, axis=1, inplace=True)

#%% Now construct factor using Turnleaf CPI data

cpitl['3m1m_rising_yoy'] = cpitl['cpi.1mf.yoy'] < cpitl['cpi.3mf.yoy']
cpitl['6m1m_rising_yoy'] = cpitl['cpi.1mf.yoy'] < cpitl['cpi.6mf.yoy']
cpitl['9m1m_rising_yoy'] = cpitl['cpi.1mf.yoy'] < cpitl['cpi.9mf.yoy']
cpitl['12m1m_rising_yoy'] = cpitl['cpi.1mf.yoy'] < cpitl['cpi.12mf.yoy']

cpitl['3m_rising_yoy'] = cpitl['cpi.realised'] < cpitl['cpi.3mf.yoy']
cpitl['6m_rising_yoy'] = cpitl['cpi.realised'] < cpitl['cpi.6mf.yoy']
cpitl['9m_rising_yoy'] = cpitl['cpi.realised'] < cpitl['cpi.9mf.yoy']
cpitl['12m_rising_yoy'] = cpitl['cpi.realised'] < cpitl['cpi.12mf.yoy']

cpitl['3m1m_rising_yoy'] = pd.Categorical(cpitl['3m1m_rising_yoy'], categories=[True, False], ordered=True)
cpitl['6m1m_rising_yoy'] = pd.Categorical(cpitl['6m1m_rising_yoy'], categories=[True, False], ordered=True)
cpitl['9m1m_rising_yoy'] = pd.Categorical(cpitl['9m1m_rising_yoy'], categories=[True, False], ordered=True)
cpitl['12m1m_rising_yoy'] = pd.Categorical(cpitl['12m1m_rising_yoy'], categories=[True, False], ordered=True)

cpitl['3m_rising_yoy'] = pd.Categorical(cpitl['3m_rising_yoy'], categories=[True, False], ordered=True)
cpitl['6m_rising_yoy'] = pd.Categorical(cpitl['6m_rising_yoy'], categories=[True, False], ordered=True)
cpitl['9m_rising_yoy'] = pd.Categorical(cpitl['9m_rising_yoy'], categories=[True, False], ordered=True)
cpitl['12m_rising_yoy'] = pd.Categorical(cpitl['12m_rising_yoy'], categories=[True, False], ordered=True)

cpitl.reset_index(drop=True, inplace=True)
cpitl.set_index('refmonthyear', inplace=True)

cpitl.to_parquet("workspace/cpi_turnleaf_factors.pq")
cpitl.to_clipboard()

# %% Add hml features to dfdaily
def get_val(df, x, col):
    if x in df.index:
        return df.loc[x][col]
    else:
        return np.NaN

dfdaily.reset_index(inplace=True)
dfreturns.reset_index(inplace=True)
dfX = dfreturns[["date", "refmonthyear", "nmi.hml", "pmi.hml", "cpi.hml"]].copy()
dfX.drop_duplicates(inplace=True)

# dfdaily.set_index('refmonthyear', inplace=True)
dfX.set_index("refmonthyear", inplace=True)

dfdaily["cpi.hml"] = dfdaily["refmonthyear"].apply(lambda x: get_val(dfX, x, "cpi.hml"))
dfdaily["nmi.hml"] = dfdaily["refmonthyear"].apply(lambda x: get_val(dfX, x, "nmi.hml"))
dfdaily["pmi.hml"] = dfdaily["refmonthyear"].apply(lambda x: get_val(dfX, x, "pmi.hml"))


dfreturns["rtn_date"] = dfreturns["date"].copy()
dfreturns.set_index(["date", "refmonthyear"], inplace=True)

# fwd_rtn.merge(right=cpi[['cpi.hml', '3M_rising', '6M_rising', '9M_rising', '12M_rising']], 
#               how='inner', left_on='refmonthyear', right_on='refmonthyear')

#%% Construct Returns DataFrame using Fred CPI factors 
fwd_rtn.reset_index(inplace=True)
monthly_cols = [f"rtn_{i+1}M" for i in range(12)]
fwd_month_returns = fwd_rtn.groupby(['refmonthyear', 'ticker']).last()[monthly_cols]
fwd_month_returns.reset_index(inplace=True)

dfreturns.reset_index(inplace=True)
cpi.reset_index(inplace=True)

dfreturns.set_index('refmonthyear', inplace=True)
cpi.set_index('refmonthyear', inplace=True)
# cpi[['refmonthyear', '3M_rising', '6M_rising', '9M_rising', '12M_rising', '3Mchg', '6Mchg', '9Mchg', '12Mchg']]


dfreturns.reset_index(inplace=True)
dfreturns.set_index(['refmonthyear', 'ticker'], inplace=True)

dfreturns.head()

fwd_month_returns.reset_index(inplace=True)
fwd_month_returns.set_index(['refmonthyear', 'ticker'], inplace=True)
fwd_month_returns.head()

dfreturns.set_index('refmonthyear', inplace=True)
for col in cpi.columns:
	if col.endswith('rising') or col.endswith('chg'):
		dfreturns[col] = cpi[col]

for col in monthly_cols:
	dfreturns[col] = fwd_month_returns[col]

dfreturns.reset_index(inplace=True)
dfreturns.to_parquet("workspace/dfreturns_with_factors.pq")


#%% Construct Returns + Factors data set using Turnleaf Derived Factors
fwd_rtn.reset_index(inplace=True)
monthly_cols = [f"rtn_{i+1}M" for i in range(12)]
fwd_month_returns = fwd_rtn.groupby(['refmonthyear', 'ticker']).last()[monthly_cols]
fwd_month_returns.reset_index(inplace=True)

dfreturns.reset_index(inplace=True)
cpitl.reset_index(inplace=True)

dfreturns.set_index('refmonthyear', inplace=True)
cpitl.set_index('refmonthyear', inplace=True)
cpitl[['3m_rising_yoy', '6m_rising_yoy', '9m_rising_yoy', '12m_rising_yoy',
     '3m1m_rising_yoy', '6m1m_rising_yoy', '9m1m_rising_yoy', '12m1m_rising_yoy']]


dfreturns.reset_index(inplace=True)
dfreturns.set_index(['refmonthyear', 'ticker'], inplace=True)

dfreturns.head()

fwd_month_returns.reset_index(inplace=True)
fwd_month_returns.set_index(['refmonthyear', 'ticker'], inplace=True)
fwd_month_returns.head()

dfreturns.reset_index(inplace=True)
dfreturns.set_index('refmonthyear', inplace=True)
for col in cpitl.columns:    
    if col.endswith('rising_yoy'):
        print(col)
        dfreturns[col] = cpitl[col]


dfreturns.reset_index(inplace=True)
dfreturns.set_index(['refmonthyear', 'ticker'], inplace=True)
for col in monthly_cols:
	dfreturns[col] = fwd_month_returns[col]

dfreturns.reset_index(inplace=True)
dfreturns.to_parquet("workspace/dfreturns_with_tlfactors.pq")



# fwd_rtn_quarterly
# cpi = pd.read_parquet("workspace/cpi_fred_factors.pq")
# fwd_rtn = pd.read_parquet("workspace/fwd_returns.pq")
# dfreturns = pd.read_parquet("workspace/returns_with_hml.pq")
# # dfreturns["date"] = dfreturns["date"].apply(lambda x: pd.to_datetime(x).date()).set_index(['date', 'refmonthyear'])
# dfreturns_wide = pd.read_parquet("workspace/monthly_returns.pq")
# daily_returns = pd.read_parquet("workspace/daily_returns.pq")
# dfdaily = pd.read_parquet("workspace/daily_levels_wide.pq")

# dfdaily = pd.read_parquet("workspace/daily_sector_levels.pq")
# dfdaily.columns = ["date", "level", "ticker", "refmonthyear"]
# dfdaily["date"] = dfdaily["date"].apply(lambda x: pd.to_datetime(x).date())
# dfdaily.set_index(["date", "refmonthyear"], inplace=True)

# Create HML dataframe, dropping return columns
# dfhml = dfreturns[["nmi.hml", "pmi.hml", "cpi.hml"]].copy().reset_index().dropna()
# dfhml = dfhml.groupby(["date", "refmonthyear"]).first()


# # %% Add hml features to dfdaily
# def get_val(df, x, col):
#     if x in df.index:
#         return df.loc[x][col]
#     else:
#         return np.NaN


# dfdaily.reset_index(inplace=True)
# dfreturns.reset_index(inplace=True)
# dfX = dfreturns[["date", "refmonthyear", "nmi.hml", "pmi.hml", "cpi.hml"]].copy()
# dfX.drop_duplicates(inplace=True)

# # dfdaily.set_index('refmonthyear', inplace=True)
# dfX.set_index("refmonthyear", inplace=True)

# dfdaily["cpi.hml"] = dfdaily["refmonthyear"].apply(lambda x: get_val(dfX, x, "cpi.hml"))
# dfdaily["nmi.hml"] = dfdaily["refmonthyear"].apply(lambda x: get_val(dfX, x, "nmi.hml"))
# dfdaily["pmi.hml"] = dfdaily["refmonthyear"].apply(lambda x: get_val(dfX, x, "pmi.hml"))

# dfreturns_wide["cpi.hml"]

