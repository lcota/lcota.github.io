# -*- coding: utf-8 -*-
#%% Imports
import os
import pandas as pd
from pandas import DataFrame, Series, DatetimeIndex
import datetime

import numpy.linalg as linalg

import matplotlib.pyplot as plt
import seaborn as sb


#%% Init
DATA_DIR = "//Volumes/RIBEIRO/data"
files = os.listdir(DATA_DIR)
files = [f for f in files if f.find('bar') > 0]
files = [f for f in files if f.find("0320") > 0]

symbols = [f.split('.')[0] for f in files]

data = {}
for fname in files:
	fpath = "{root}/{fname}".format(root = DATA_DIR, fname = fname)
	symbol = fname.split('.')[0]
	df = pd.read_csv(fpath, index_col = 'times')
	df[symbol] = df['close']	
	df = df[symbol]
	data[symbol] = df

#%% Clean dataset, remove nans
df = DataFrame(data = data)
df.dropna(inplace = True)

#%% calculate returns
returns = df / df.shift(1) - 1.0
returns.dropna(inplace = True)

#%% PCA intermediate calculations
means = returns.mean()
cov = returns.cov()
eigvals, eigvecs = linalg.eig(cov)
print cov
print eigvals
print eigvecs

#sb.heatmap(cov)

## now compute eig decomp on correlation matrix
corrmat = returns.corr()
eigvals, eigvecs = linalg.eig(corrmat)

print corrmat
print eigvals
print eigvecs

#sb.heatmap(corrmat)

#%% PCA Decomposition with SKLearn
from sklearn.decomposition import PCA
pca = PCA()
yhat = pca.fit_transform(X = corrmat)
