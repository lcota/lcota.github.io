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


def make_fixed_weights(dict_weights, returns):
    weights = pd.DataFrame(dict_weights, index=returns.index)
    return weights


def make_vol_weights(returns, window=30): #, max_leverage=1, min_leverage=0):
    """
    Calculates a naive vol-contribution per ETF such that equal amounts are allocated
    to each unit of risk. Higher vol assets will have lower allocation than lower vol 
    assets. 
    
    """
    weights = (returns.pct_change()
             .rolling(window=window, min_periods=window)
             .std()
             .dropna()
             .apply(lambda x: np.round(x * 100 * np.sqrt(252), 2)))
    weights = 1. / weights
    # weights = returns.rolling(window=lookback, min_periods=lookback).apply(lambda x: max_leverage * x / x.sum())
    gross_vol = weights.apply('sum', axis=1)
    weights = weights.div(gross_vol, axis=0)
    return weights


def make_index(returns, weights):
    min_date = max(returns.index.min(), weights.index.min())
    
    df = (weights[weights.index >= min_date] * returns[returns.index >= min_date]).sum(axis=1)
    # df = (weights * df_).sum(axis=1)

    return df

def calc_drawdowns(strategy_returns, period='1M'):
    """
    Calculate drawdowns of a strategy given a series of returns.
    """
    df_drawdowns = strategy_returns.resample(rule=period).apply('ohlc')
    df_drawdowns['drawdown'] = 100 * (df_drawdowns['close'] / df_drawdowns['open'] - 1)
        
    return df_drawdowns['drawdown']

def calc_vol(strategy_returns, period='1M'):
    """
    Calculate volatility of a strategy given a series of returns.
    """
    df_vol = strategy_returns.resample(rule=period).apply('ohlc')
    df_vol = (df_vol['close'] / df_vol['open'] - 1).std() * np.sqrt(252)
    # df_vol = df_vol * np.sqrt(252)
        
    return df_vol


def calc_ir(strategy_returns, benchmark_returns, period='1M'):
    """
    Calculate information ratio of a strategy given a series of returns.
    """
    mean_strat_returns = strategy_returns.resample(rule=period).apply('ohlc')
    mean_strat_returns = mean_strat_returns['close'] / mean_strat_returns['open'] - 1
    
    mean_benchmark_returns = benchmark_returns.resample(rule=period).apply('ohlc')
    mean_benchmark_returns = mean_benchmark_returns['close'] / mean_benchmark_returns['open'] - 1
    
    df_ir = mean_strat_returns / mean_benchmark_returns.apply('std')
    
    return df_ir


def trade_signal(signal, returns, benchmark=None):
    root_name = returns.name
    # base_trcol = f"{root_name}.tr"
    # sigwgt_trcol= f"{root_name}.sigwgt.tr"    
    base_trcol = f"base_tr"
    sigwgt_trcol= f"sigwgt.tr"
    
    df = pd.merge(left=signal, right=returns, 
                  left_index=True, right_index=True, how='inner')
    
    # base_returns = 100 * (1.0 + df[root_name]).cumprod()
    # df = pd.DataFrame(returns)
    # df['sig'] = signal
    df['base.rtn'] = df[returns.name]
    df['wgt.rtn'] = df[returns.name] * df[signal.name]
    df[base_trcol] = (1 + df[returns.name]).cumprod() * 100
    df[sigwgt_trcol] = (1 + df['wgt.rtn']).cumprod() * 100    
    df.dropna(inplace=True)    
    return df
