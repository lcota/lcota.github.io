# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 11:43:57 2024

@author: lcota
"""
#%% import statements
# %load_ext autoreload
# %autoreload 2

#%% imports 
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

import features
from features import dfx, dfy, dfref

#%% Weighted combination signal
wgtpmi = -.25
wgtnmi = 1.5
wgtcpi = -.25
dfx['wgt.combo.signal'] = wgtpmi * dfx['pmi.above'] + wgtnmi * dfx['nmi.above'] + wgtcpi * dfx['cpi.above']


#%% plot strategy returns
signames = ['pmi.above', 'nmi.above', 'cpi.above', 'wgt.combo.signal']
ticker = 'spy'
lag = "na"
prefix = "trailing_thresh"
shorts = "noshort"
strats = []

for ticker in dfy.columns:
    returns = dfy[ticker]
    print("===================================")
    print(ticker)
    print("-------")
    for signame in signames[1:2]:
        figtitle = f"{prefix}.{signame}.{ticker}.lag-{lag}.{shorts}"        
        signal = dfx[signame]       
        
        dfstrat = trade_signal(signal, returns)        
        dfstrat['ticker'] = ticker
        dfstrat['signame'] = signame        
        
        fig, ax = plt.subplots(figsize=(14, 8))
        dfstrat[dfstrat.columns[-4:]].plot(title=f"{figtitle}", ax=ax)
        plt.grid(ls='--', color='grey', alpha=.5)
        
        figname = f"plots/{figtitle}.png"
        # print(figname)
        # plt.savefig(figname, dpi=200)
        # plt.close()
        
        strats.append(dfstrat[['ticker', 'signame', 'base.rtn', 'wgt.rtn']])

strats = pd.concat(strats)

#%% scatter plots - 3 month changes


#%% strat statistics (IR, STDev)
def vol(x):
    return np.std(x) * np.sqrt(12)

def annrtn(x):
    return np.mean(x) * np.sqrt(12)

def sharpe(x):
    return np.mean(x) / np.std(x)

strat_stats = (strats.groupby(['ticker', 'signame'])
               .agg([vol, annrtn, sharpe]))

strat_stats['ir'] = strat_stats['wgt.rtn', 'annrtn'] / strat_stats['base.rtn', 'vol']

head(strat_stats)
strat_stats.to_clipboard()

#%% optimize weights for combination signal
ticker = '6040'
indicators = dfx[['pmi.above', 'nmi.above', 'cpi.above']]
returns = dfy[ticker]


def optim_weights(indicators, 
                  returns, 
                  min_return = .08,
                  target_volatility=.15, 
                  solver='SCS',
                  verbose=False):

    df = pd.merge(left=indicators, right=returns, 
                  left_index=True, right_index=True, how='inner')
    
    
    asset_returns = df[returns.name]  
    asset_returns = np.reshape(asset_returns, newshape=(asset_returns.shape[0], 1))
    economic_indicators = df[indicators.columns].T
    target_volatility = target_volatility  
    
    print(f"{economic_indicators.shape} vs {asset_returns.shape}")
    
    expected_returns = economic_indicators @ asset_returns
    weights = cp.Variable(len(indicators.columns))
    
    objective = cp.Maximize(weights @ expected_returns)
    gamma = cp.Parameter(nonneg=True)
    
    
    # Define the constraints
    constraints = [
        cp.sum(weights) == 1,  # Sum of weights must be 1
        cp.norm(weights @ economic_indicators) <= target_volatility,
        # (weights @ expected_returns) >= min_return,
        weights >= 0,
        # weights <= 1
    ]
    
    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)
    
    # Output the optimal weights
    optimal_weights = weights.value
    print("Optimal Weights:", optimal_weights)

    return problem




#%%
ticker = 'spy'
returns = dfy[ticker]
indicators = dfx[['pmi.above', 'nmi.above', 'cpi.above']]
solver='SCS'
# 'CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCIPY', 'SCS'
min_return = .03
target_volatility  = .25

soln = optim_weights(indicators=indicators, 
                     returns=returns, 
                     solver=solver,
                     min_return=min_return,
                     target_volatility=.15,
                     verbose=True)

soln.status

print(cp.installed_solvers())

#%% new version optim function

# ticker = ['spy', 'agg']
returns = dfy.copy()
indicators = dfx[['pmi.above', 'nmi.above', 'cpi.above']].copy()
df = pd.merge(left=indicators, right=returns, left_index=True, right_index=True, how='inner')

returns = df[returns.columns]
indicators = df[indicators.columns]

f_loadings = np.zeros((indicators.shape[1], returns.shape[1]))
for i in range(returns.shape[1]):
    y = returns.iloc[:, i]
    X = sm.add_constant(indicators)
    model = sm.OLS(y, X).fit()
    f_loadings[:, i] = model.params[1:]


max_leverage = 2.0
gamma_value = 0.1

m = len(indicators.columns) # number of factors
n = len(dfy.columns) # number of assets


rtn = df[returns.columns]
mu = rtn.mean()
mu = np.reshape(mu, (mu.shape[0], 1))
Sigma = rtn.T.dot(rtn)
Sigma_tilde = indicators.T.dot(indicators)
D = sp.diags(np.random.uniform(0, 0.9, size=n))
F = f_loadings # factor loading matrix

#  factor model portfolio optimization
w = cp.Variable((n, 1))
f = cp.Variable((m, 1))
# f = cp.reshape(f, shape=(f.shape[0], 1))
gamma = cp.Parameter(nonneg=True)
Lmax = cp.Parameter()
ret = mu.T @ w

Lmax.value = max_leverage
gamma.value = gamma_value

risk = cp.quad_form(f, Sigma_tilde) + cp.sum_squares(np.sqrt(D) @ w)
objective = cp.Maximize(ret - gamma*risk)
constraints = [
    cp.sum(w) == 1,
    # cp.norm(w @ F) <= target_volatility,
    # cp.norm(F.T @ w),
    f == F.T @ w,
    cp.norm(w, 1) <= Lmax,
    ]

prob_factor = cp.Problem(objective, constraints)

prob_factor.solve(verbose=False)
print(prob_factor)

f.value
w.value
F
f_loadings




#%% Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

x_cols = [c for c in dfx.columns if 'above' in c] + [c for c in dfx.columns if '.diff' in c]
rfresults = []
rfmodels = {}
for y_target in dfy.columns:
    
    Y = dfy[y_target] > 0
    X = dfx[x_cols]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Create an instance of Logistic Regression Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the parameters    
    # Train the model using the training sets
    rf.fit(X_train, Y_train)
    rfmodels[y_target] = rf
    
    # Predicting the Test set rfresults
    Y_pred = rf.predict(X_test)
    
    # Calculate the accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    pct_up = sum(Y > 0) / len(Y)
    
    rfresults.append([y_target.upper(), np.round(accuracy, 2), np.round(pct_up, 2), np.round(accuracy - pct_up, 2)])
    print(f'{y_target} Model Accuracy vs Observed: \t {accuracy:.2f} / {pct_up:.2f}')
    # print(f'{y_target} Observed Up: {pct_up:.2f}')

rfresults = pd.DataFrame(rfresults, columns=['ticker', 'accuracy', 'observed', 'improvement'])
rfresults.set_index('ticker', inplace=True)
rfresults['name'] = dfref['name']
rfresults.sort_values(by='improvement', ascending=False, inplace=True)

rftbl = []
for ticker, model in rfmodels.items():
    coefs = model.feature_importances_.reshape((1, len(model.feature_importances_)))
    tbl = make_coef_table(ticker.upper(), coefs, x_cols)    
    rftbl.append(tbl)

rftbl = pd.concat(rftbl)
rftbl.set_index('ticker', inplace=True)
rftbl = pd.merge(rfresults, rftbl, left_index=True, right_index=True)
rftbl.to_clipboard()

rfresults.to_clipboard()




















