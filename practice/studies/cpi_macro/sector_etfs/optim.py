#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:28:24 2024

@author: lcota
"""
#%% Imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
# import modin.pandas as pd
# import modin
import pandas as pd
from scipy.optimize import minimize 
from utils import *


#%% Load Data
dfreturns = pd.read_parquet("workspace/dfreturns_with_factors.pq")
factor_cols = ['date', 'refmonthyear', 'cpi.hml', 'pmi.hml', 'nmi.hml', 
                '3M_rising', '6M_rising', '9M_rising', '12M_rising']


#%% Optimization Helper Functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std_dev

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    p_returns, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std_dev

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0, short_size=0, long_size=1):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((short_size, long_size) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def calculate_factor_exposures(returns, factors):
    factor_exposures = {}
    for stock in returns.columns:
        exposures = []
        for factor in factors.columns:
            # Calculate the mean return for each category of the factor
            grouped_returns = returns[stock].groupby(factors[factor]).mean()
            exposures.append(grouped_returns.mean())
        factor_exposures[stock] = exposures
    return pd.DataFrame(factor_exposures, index=factors.columns)

def calculate_factor_exposures2(returns, factors):
    factor_exposures = {}
    stocks = []
    for stock in returns.columns:
        stocks.append(stock)
        exposures = {}
        for factor in factors.columns:
            # Calculate the mean return for each category of the factor
            grouped_returns = returns[stock].groupby(factors[factor]).mean()
            print(grouped_returns.index)
            for v in grouped_returns.index.values:
                factor_flattened = f"{factor}_{v}"
                exposures[factor_flattened] = grouped_returns[v]
        # exposures = pd.DataFrame(exposures)
        factor_exposures[stock] = exposures
    
    factor_cols = factor_exposures[stocks[0]].keys()
    print(factor_cols)
    factor_exposures = pd.DataFrame(factor_exposures, index=factor_cols)
    return factor_exposures


#%% Windowed Optimization Function 
factor_cols = [ # 'cpi.hml', 
                # 'pmi.hml', 
                # 'nmi.hml', 
                '3M_rising', 
                '6M_rising', 
                '9M_rising', 
                '12M_rising']

rtn_col = 'rtn_3M'
lookback_periods = 12
use_excess_returns=False
short_size=0
long_size=0

# def calc_strat_returns(dfreturns, factor_cols=None, rtn_col='rtn_3M', lookback_periods=12, short_size=0, long_size=1, use_excess_returns=False):
dfreturns.set_index(['date', 'refmonthyear'], inplace=True)

factors = dfreturns[factor_cols].copy()
# factors.set_index(['date', 'refmonthyear'], inplace=True)

dfreturns.reset_index(inplace=True)
monthly_returns = dfreturns.groupby(['refmonthyear', 'ticker'])[['date', rtn_col]].last().reset_index()
monthly_returns = monthly_returns.pivot(columns='ticker', index=['date', 'refmonthyear'], values=rtn_col)

market_return = monthly_returns['spx'].copy()
excess_returns = monthly_returns.drop('spx', axis=1).copy()
sector_returns = monthly_returns.drop('spx', axis=1).copy()

# Calculate excess returns
tickers = [t for t in dfreturns.ticker.unique() if t != 'spx']
for t in tickers:
    excess_returns[t] -= market_return

returns = excess_returns if use_excess_returns else sector_returns


#%% Estimation loop
weights_history = []
portfolio_returns_history = []

i = 12
# for i in range(lookback_periods, len(returns)):
# Select the lookback window
returns_window = returns.iloc[i - lookback_periods:i]
factors_window = factors.iloc[i - lookback_periods:i]

# Calculate factor exposures for the lookback window

factor_exposures = calculate_factor_exposures2(returns_window, factors_window)
# Calculate mean and covariance of stock returns
mean_factor_returns = pd.DataFrame(factor_exposures.mean(axis=0))
cov_factor_returns = factor_exposures.cov()
mean_stock_returns = factor_exposures @ mean_factor_returns

print(mean_factor_returns.shape, mean_stock_returns.shape, factor_exposures.T.shape, cov_factor_returns.shape)
print(mean_stock_returns)
print(factor_exposures)
print(factor_exposures.T)
print(cov_factor_returns)
cov_matrix_stock_returns = factor_exposures @ cov_factor_returns.T @ factor_exposures


# mean_stock_returns = factor_exposures.T @ mean_factor_returns
# cov_matrix_stock_returns = returns_window.cov()

# Perform optimization to get weights
weights = optimize_portfolio(mean_stock_returns, cov_matrix_stock_returns, short_size=short_size, long_size=long_size)
weights_history.append(weights)

# Calculate the portfolio return for the next period
next_period_return = np.dot(weights, returns.iloc[i])
portfolio_returns_history.append(next_period_return)


# return retvals