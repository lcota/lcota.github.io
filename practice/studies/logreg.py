# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:58:56 2024

@author: lcota
"""
#%% imports 
from importlib import reload
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

import datatable as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as npt
import statsmodels.api as sm

import datatools
from datatools import *
import strategy
from strategy import *
import features
from features import dfx, dfy, dfref

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#%% util functions for analyzing model results
def make_coef_table(ticker, coefs, colnames):
    tbl = pd.DataFrame(coefs, columns=colnames)
    tbl['ticker'] = ticker
    return tbl

#%% Rolling regression training function
def train_lr_rolling(X, Y, window=36):
    results = {}
    models = {}
    common_dates = X.index.intersection(Y.index)
    _X = X.loc[common_dates]
    _Y = Y.loc[common_dates]
    N = len(_Y.index)
    
    act = []
    preds = []
    probUp = []
    probDn = []
    lstrades = []
    lotrades = []
    models = {}
    
    for i in range(window, N):
        mdl = LogisticRegression()
        mdl.fit(_X[i-window: i-1], _Y[i-window: i-1])
        x = _X.iloc[i].values.reshape(1, -1)
        pred = mdl.predict(x)        
        ls = 1 if pred[0] == True else -1
        lo = 1 if pred[0] == True else 0
        
        act.append(_Y.iloc[i])
        preds.append(pred[0])
        probs = mdl.predict_proba(x)
        probDn.append(probs[0][0])
        probUp.append(probs[0][1])
        lstrades.append(ls)
        lotrades.append(lo)
        
        models[_X.index[i]] = mdl
    
    data = {'y': act, 
            'y_hat': preds,
            'probDn': probDn,
            'probUp': probUp,
            'ls': lstrades,
            'lo': lotrades}
    df = pd.DataFrame(data=data, index=_X.index[window:])
    return df, models


models = {}
results = []
dfresults = None
dfy_lag1 = lag(dfy)

# x_cols = [c for c in dfx.columns if 'above' in c] + [c for c in dfx.columns if '.diff' in c]
# reduced_x_cols = ['pmi1m.above','pmi6m.above','nmi1m.above','nmi3m.above','nmi9m.above','cpiyoy1m.diff','cpiyoy3m.diff','pmi1m.diff','pmi6m.diff','nmi3m.diff','nmi12m.diff']
reduced_x_cols = ['pmi1m.above','pmi6m.above','nmi1m.above','nmi3m.above','nmi9m.above','pmi1m.diff','pmi6m.diff','nmi3m.diff','nmi12m.diff']
x_cols = reduced_x_cols

dfX = dfx[x_cols]
y_target = 'spy'
for y_target in dfy.columns:
    dfY = dfy[y_target] > 0
    try:
        dfres, mdls = train_lr_rolling(dfX, dfY)
        dfres['ticker'] = y_target
        models[y_target] = mdls
        results.append(dfres)
    except:
        print(f"Error with {y_target}")

dfresults = pd.concat(results)
dfresults.to_parquet("data/results/results_ex_cpi.pq")

#%% Training logistic regression
penalty_ = 'l2'
direction = 'up'
suffix = 'full'

# Train reduced model using coefficients where in aggregate, z.test has p-value of 10% significance
reduced_x_cols = ['pmi1m.above','pmi6m.above','nmi1m.above','nmi3m.above','nmi9m.above','cpiyoy1m.diff','cpiyoy3m.diff','pmi1m.diff','pmi6m.diff','nmi3m.diff','nmi12m.diff']
x_cols = [c for c in dfx.columns if 'above' in c] + [c for c in dfx.columns if '.diff' in c]
# x_cols = reduced_x_cols

lrresults = []
lrmodels = {}
for y_target in dfy.columns:
    Y = dfy[y_target] > 0
    X = dfx[x_cols]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Create an instance of Logistic Regression Classifier
    
    log_reg = LogisticRegression(penalty=penalty_, C=1.0)
    
    # Train the model using the training sets
    log_reg.fit(X_train, Y_train)
    lrmodels[y_target] = log_reg
    
    # Predicting the Test set logregresults
    Y_pred = log_reg.predict(X_test)
    
    # Calculate the accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    pct_obs = sum(Y == True) / len(Y)    
    
    lrresults.append([y_target.upper(), 
                      direction,
                      np.round(accuracy, 2), 
                      np.round(pct_obs, 2), 
                      np.round(accuracy - pct_obs, 2)])    
    # print(f'{y_target.upper()} Model Accuracy vs Observed: \t {accuracy:.2f} / {pct_obs:.2f} / {accuracy - pct_obs:.2f}')
    # print(f'{y_target} Observed Up: {pct_up:.2f}')

lrresults = pd.DataFrame(lrresults, columns=['ticker', 'direction', 'accuracy', 'observed', 'improvement'])
lrresults.set_index('ticker', inplace=True)
lrresults = pd.merge(left=lrresults, right=dfref[['name', 'assetclass', 'subgroup', 'style', 'maturity', 'duration']], left_index=True, right_index=True)
lrresults = lrresults[['name', 'assetclass', 'subgroup', 'style', 'maturity', 'duration', 
                       'direction', 'accuracy', 'observed', 'improvement']]
lrresults.sort_values(by='improvement', ascending=False)

lrtbl = []
for ticker, model in lrmodels.items():
    tbl = make_coef_table(ticker.upper(), model.coef_, x_cols)
    lrtbl.append(tbl)
lrtbl = pd.concat(lrtbl)
lrtbl.set_index('ticker', inplace=True)
lrtbl = pd.merge(lrresults, lrtbl, left_index=True, right_index=True)
lrtbl.sort_values('improvement', ascending=False, inplace=True)

#%% Coefficient Analysis
coefbyasset = lrtbl.groupby("assetclass")[x_cols].agg(['mean', 'std'])
coefbyasset
coefmeans_by_asset.to_clipboard()

coefmeans = lrtbl[x_cols].mean()

#%% Save logistic regression model coefs
filename = f"data/models/lr{suffix}_{direction}_{penalty_}.csv"
lrtbl.to_csv(filename, index=False)



#%% plots
for c in reduced_x_cols:
    figtitle = f"{c} by asset class"
    sb.boxplot(data=lrtbl, y=c, hue='assetclass')
    figname = f"plots/feature_plots/{figtitle}.boxplot.png"
    plt.savefig(figname, dpi=200)
    plt.close()
