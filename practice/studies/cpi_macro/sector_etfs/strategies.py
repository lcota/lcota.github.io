#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:24:21 2024

@author: lcota
"""
# import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
# import modin.pandas as pd
# import modin
import pandas as pd
from scipy.optimize import minimize 
from utils import *

from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont("/Users/lcota/Library/Fonts/Raleway-Regular.ttf")
fm.fontManager.addfont("/Users/lcota/Library/Fonts/Raleway-Light.ttf")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Raleway'
plt.rcParams['font.style'] = "normal"

plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titlecolor'] = '#595959'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelcolor'] = '#595959'
plt.rcParams['xtick.color'] = '#595959'
plt.rcParams['ytick.color'] = '#595959'

def calc_excess_returns(df, factor_col):
    """Returns the excess monthly returns for 1M-12M forward periods. Also returns
    T-test results for significance of excess return estimates."""
    mkt_returns = df[df.ticker=='spx'].copy()

    sector_tickers = [x for x in df.ticker.unique() if x != 'spx']
    sector_returns = df[df.ticker.isin(sector_tickers)]

    cols = [c for c in df.columns if c.endswith("M")]
    factor_returns = df.groupby(['ticker', factor_col])[cols].agg('mean').unstack()
    excess_returns = factor_returns - factor_returns.loc['spx']

    return factor_returns, excess_returns


def factor_stats(df, factor_col, rtn_col, use_excess_returns=True, 
                 show_plot=True, save_plot=False, 
                 title=None, xlabel=None, ylabel=None, figsize=None):
    pass
    


def factor_barplot(df, factor_col, rtn_col, use_excess_returns=True, show_plot=True, 
                   save_plot=False,
                   title=None, xlabel=None, ylabel=None, figsize=None):
    # fig, ax = plt.subplots(figsize=(10,5)) 
    plt.ioff()

    # plt.figure(figsize=(10,5))
    # sb.barplot()

    if figsize is not None:
        plt.rcParams['figure.figsize'] = figsize

    mean_returns = df[df[factor_col].isna() == False].groupby(['ticker', factor_col], observed=True)[rtn_col].agg('mean')
    excess_returns = mean_returns - mean_returns['spx']
    excess_returns = excess_returns.reset_index()


    returns = mean_returns
    if use_excess_returns:
        returns = excess_returns
        if title is None:
            title = f"Excess Sector Returns {rtn_col} x {factor_col}"
    else:
        returns = mean_returns.reset_index()
        if title is None:
            title = f"Sector Returns {rtn_col} x {factor_col}"

    returns = pd.DataFrame(returns)
    returns[rtn_col] = returns[rtn_col] * 100 #convert to pct scale

    plt.figure(figsize=(10,5))
    sb.barplot(returns, x='ticker', y=rtn_col, hue=factor_col)
    # ax.set_figure(figsize=(10,5))
    plt.grid(ls='--', alpha=.5)
    plt.xticks(rotation=45)
    plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)


    figname = f"img/excess_return_{factor_col}_{rtn_col}.png"
    if use_excess_returns == False:
        figname = f"img/total_return_{factor_col}_{rtn_col}.png"
    if save_plot:
        # print(f"Saving {figname}")
        plt.savefig(figname)

    if show_plot:
        plt.show()





# %% Test trading rules
# For each individual ticker in dataframe, what is the best way to position
# either long/short/neutral, conditional on each individual regime? Output should be
# 3 "strategies" per ticker, one for each of cpi.hml, pmi.hml and nmi.hml

dfreturns = pd.read_parquet("workspace/dfreturns_with_factors.pq")
factor_cols = ['date', 'refmonthyear', 'cpi.hml', 'pmi.hml', 'nmi.hml', 
                '3M_rising', '6M_rising', '9M_rising', '12M_rising']
# Generate figures for factors x fwd return periods
def gen_factor_plots(dfreturns, factor_cols, monthly_cols, use_excess_returns=True):
    monthly_cols = [f"rtn_{i+1}M" for i in range(12)]
    for mc in monthly_cols:
        for fc in factor_cols:
            if fc not in ['date', 'refmonthyear']:
                if fc.endswith('rising'):
                    dfreturns[fc] = pd.Categorical(values=dfreturns[fc], categories=[True, False], ordered=True)
                factor_barplot(dfreturns, factor_col=fc, rtn_col=mc, use_excess_returns=use_excess_returns, show_plot=False, save_plot=True)


# Method 1: Average 1M/3M/6M/9M/12M returns given CPI HML

# Mean-Variance Optimization
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
    factor_exposures = pd.DataFrame(index=factors.columns, columns=returns.columns)
    for stock in returns.columns:
        for factor in factors.columns:
            # Calculate the mean return for each category of the factor
            grouped_returns = returns[stock].groupby(factors[factor]).mean()
            factor_exposures.at[factor, stock] = grouped_returns.mean()
    return factor_exposures.astype(float)


# Rolling window optimization
def calc_strat_returns(dfreturns, factor_cols=None, rtn_col='rtn_3M', 
                       lookback_periods=12, short_size=0, long_size=1, 
                       use_excess_returns=False):
    if factor_cols is None:
        factor_cols = ['date', 'refmonthyear', 'cpi.hml', 'pmi.hml', 'nmi.hml', 
                        '3M_rising', '6M_rising', '9M_rising', '12M_rising']

    dfreturns.set_index(['date', 'refmonthyear'], inplace=True)

    factors = dfreturns[factor_cols].drop_duplicates()
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


    weights_history = []
    portfolio_returns_history = []


    for i in range(lookback_periods, len(returns)):
        # Select the lookback window
        returns_window = returns.iloc[i - lookback_periods:i]
        factors_window = factors.iloc[i - lookback_periods:i]
        
        # Calculate factor exposures for the lookback window
        # factor_exposures = calculate_factor_exposures(returns_window, factors_window)
        factor_exposures = calculate_factor_exposures(returns_window, factors_window)
        # Calculate mean and covariance of stock returns
        mean_factor_returns = pd.DataFrame(factor_exposures.mean(axis=1))
        cov_factor_returns = factor_exposures.cov()
        mean_stock_returns = factor_exposures.T @ mean_factor_returns

        print(mean_factor_returns.shape, mean_stock_returns.shape, factor_exposures.T.shape, cov_factor_returns.shape)
        print(mean_stock_returns)
        print(factor_exposures)
        print(factor_exposures.T)
        print(cov_factor_returns)
        cov_matrix_stock_returns = factor_exposures.T @ cov_factor_returns @ factor_exposures.T
    

        # mean_stock_returns = factor_exposures.T @ mean_factor_returns
        # cov_matrix_stock_returns = returns_window.cov()
        
        # Perform optimization to get weights
        weights = optimize_portfolio(mean_stock_returns, cov_matrix_stock_returns, short_size=short_size, long_size=long_size)
        weights_history.append(weights)
        
        # Calculate the portfolio return for the next period
        next_period_return = np.dot(weights, returns.iloc[i])
        portfolio_returns_history.append(next_period_return)


    # Convert weights history to DataFrame
    weights_history_df = pd.DataFrame(weights_history, index=returns.index[lookback_periods:], columns=returns.columns)

    # Convert portfolio returns history to Series
    portfolio_returns_series = pd.Series(portfolio_returns_history, index=returns.index[lookback_periods:])

    # Calculate cumulative returns
    market_cum_returns = pd.DataFrame((1 + market_return).cumprod() * 100)
    market_cum_returns.reset_index(inplace=True)

    cumulative_returns = (1 + portfolio_returns_series).cumprod() * 100
    cumulative_returns = pd.DataFrame(cumulative_returns)
    cumulative_returns.columns = ['strat_cum_rtn']
    cumulative_returns.reset_index(inplace=True)
    cumulative_returns['spx'] = market_cum_returns['spx']

    retvals = {
        'weights': weights_history_df,
        'port_returns': portfolio_returns_series,
        'cum_returns': cumulative_returns
    }
    return retvals

if __name__ == '__main__':
    factor_cols = [ # 'cpi.hml', 
                    # 'pmi.hml', 
                    # 'nmi.hml', 
                    '3M_rising', 
                    '6M_rising', 
                    '9M_rising', 
                    '12M_rising']

    rtn_col = 'rtn_3M'
    ret = calc_strat_returns(dfreturns, factor_cols=factor_cols, rtn_col=rtn_col)
    weights = ret['weights']
    cumulative_returns = ret['cum_returns']
    title = f"Cum Returns of Long Only MVO Port vs SPX\n {rtn_col} ~ 3m-12m rising factors"

    # cumulative_returns.plot(title='Cumulative Total Return of the Portfolio')
    sb.lineplot(cumulative_returns, x='date', y='strat_cum_rtn', label='MVO Portfolio')
    sb.lineplot(cumulative_returns, x='date', y='spx', alpha=.5, color='grey', ls='--', label='SPX')
    plt.legend()
    plt.grid(alpha=.5, ls='--')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title(title)
    plt.show()


    ret.keys()





# Method 2: Test CPI Rising/Falling per ticker


# betas, models = calculate_betas(dfreturns_wide, benchmark="spx", label="all")


# m = models["s5tech"]
# m.summary()
