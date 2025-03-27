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

dfx = dfx[['refmonthyear', 'cpiyoy', 'pmnmi', 'pmpmi', 'cpiyoy.surprise']]
dfx.set_index('refmonthyear', inplace=True)
dfy.set_index('refmonthyear', inplace=True)

#################################################################
#%% create signal
pmi_thresh = 50
# cpi_thresh = dfx['cpiyoy'].median()
cpi_thresh = roll_quantile(dfx['cpiyoy'], colname='cpiyoy.q50')

dfx['cpi.thresh'] = cpi_thresh

dfx['pmi.above'] = (dfx['pmpmi'] > pmi_thresh).astype(int)
dfx['nmi.above'] = (dfx['pmnmi'] > pmi_thresh).astype(int)
# dfx['cpi.above'] = (dfx['cpiyoy'] > cpi_thresh).astype(int)
dfx['cpi.above'] = (dfx['cpiyoy'] > dfx['cpi.thresh']).astype(int)

dfx.dropna(inplace=True)
dfy.dropna(inplace=True)


returns = dfy.copy()
indicators = dfx[['pmi.above', 'nmi.above', 'cpi.above']].copy()
df = pd.merge(left=indicators, right=returns, left_index=True, right_index=True, how='inner')

returns = df[returns.columns]
indicators = df[indicators.columns]

F = np.zeros((indicators.shape[1], returns.shape[1]))
for i in range(returns.shape[1]):
    y = returns.iloc[:, i]
    X = sm.add_constant(indicators)
    model = sm.OLS(y, X).fit()
    F[:, i] = model.params[1:]

print(pd.DataFrame(F.T, columns=indicators.columns, index=returns.columns))




# df = pd.merge(left=indicators, right=returns, 
#               left_index=True, right_index=True, how='inner')


# asset_returns = df[returns.name]  
# asset_returns = np.reshape(asset_returns, newshape=(asset_returns.shape[0], 1))
# economic_indicators = df[indicators.columns].T
# target_volatility = target_volatility  

# print(f"{economic_indicators.shape} vs {asset_returns.shape}")

# expected_returns = economic_indicators @ asset_returns
# weights = cp.Variable(len(indicators.columns))

# objective = cp.Maximize(weights @ expected_returns)
# gamma = cp.Parameter(nonneg=True)


# # Define the constraints
# constraints = [
#     cp.sum(weights) == 1,  # Sum of weights must be 1
#     cp.norm(weights @ economic_indicators) <= target_volatility,
#     # (weights @ expected_returns) >= min_return,
#     weights >= 0,
#     # weights <= 1
# ]

# # Define and solve the problem
# problem = cp.Problem(objective, constraints)
# problem.solve(solver=solver, verbose=verbose)

# # Output the optimal weights
# optimal_weights = weights.value
# print("Optimal Weights:", optimal_weights)