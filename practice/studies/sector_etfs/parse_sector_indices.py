#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:37:57 2024

@author: lcota
"""
import os
import pandas as pd
import datetime

root = '/Users/lcota/Library/CloudStorage/GoogleDrive-luis.cota@turnleafanalytics.com/Shared drives/turnleaf_projects/luis_cota/Sector ETFs/'
files = os.listdir("/Users/lcota/Library/CloudStorage/GoogleDrive-luis.cota@turnleafanalytics.com/Shared drives/turnleaf_projects/luis_cota/Sector ETFs/")
files = [x for x in files if x.endswith('csv')]
files = [x for x in files if len(x) > 8]
files.append("SPX.csv")

files

indices = []

fpath = "{0}{1}".format(root, files[0])

def read_etf_file(fpath):
    df = pd.read_csv(fpath)
    
    df.columns = [x.lower() for x in df.columns]
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    trcol = [x for x in df.columns if x.find('gross_dvd') > 0][0]
    ticker = trcol.split('.')[0]
    trname = 'tot_rtn_gross_dvd'
    cols = ['date', trcol]
    df = df[cols]
    cols = ['date', trname]
    df.columns = cols
    df['ticker'] = ticker
    df.set_index('date', inplace=True)
    return df

nmi = pd.read_csv(f"{root}/{files[1]}")
nmi['date'] = nmpmi['date'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

pmi = pd.read_csv(f"{root}/{files[-2]}")
pmi['date'] = pmi['date'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

nmi['refmonthyear'] = nmi['date'].apply(lambda x: x.strftime('%b.%Y').upper())
pmi['refmonthyear'] = pmi['date'].apply(lambda x: x.strftime('%b.%Y').upper())
nmi.drop(['nmpmi_release', 'nmpmi_release_date'], axis=1, inplace=True)
pmi.drop(['pmi_actual', 'pmi_release_date'], axis=1, inplace=True)

nmi.columns = ['date', 'nmi', 'refmonthyear']
nmi.to_parquet("sector_etfs/workspace/ism_nmi_full.pq")
pmi.to_parquet("sector_etfs/workspace/ism_pmi_full.pq")


ism_df = pd.merge(pmi[['date', 'pmi']], nmpmi[['date', 'nmpmi']], left_on='date', right_on='date', how='inner')
ism_df.head()
ism_df.tail()

ism_df.columns = ['date', 'pmi', 'nmi']
ism_df['refmonthyear'] = ism_df['date'].apply(lambda x: x.strftime("%b.%Y").upper())
ism_df.to_parquet("sector_etfs/workspace/ism_df.pq")

#%% read & clean SP Sector index files
index_files = [f for f in files if f.startswith('ISM') == False]
index_returns = [read_etf_file(f"{root}/{fname}") for fname in index_files]
dfrtn = pd.concat(index_returns)

dfrtn.reset_index(inplace=True)
dfrtn['refmonthyear'] = dfrtn['date'].apply(lambda x: x.strftime("%b.%Y").upper())
dfrtn.set_index('date', inplace=True)

# dfrtn = dfrtn.reset_index().pivot(columns='ticker', index=['date', 'refmonthyear'], values='tot_rtn_gross_dvd')

dfrtn.to_parquet("sector_etfs/workspace/sector_returns.pq")


