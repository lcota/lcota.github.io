# -*- coding: utf-8 -*-
import statsmodels.api as sm
import pandas as pd
import seaborn as sb
import numpy as np
import arcticdb as adb

__ac = None
etfdb = None

if __ac is None:
    __ac = adb.Arctic("lmdb:///Users/lcota/Dropbox/prj/Turnleaf_Analytics/sector_etfs/workspace.db?map_size=5GB")
if etfdb is None:
    # etfdb = __ac.create_library("etfdb")
    etfdb = __ac['etfdb']



def head(df, n=5):
    return df[0:n]


def tail(df, n=5):
    return df[-n:]


def lag(df, n=1):
    """Assumes df is sorted in ascending order by date"""
    return df.shift(n)


# %% HML function definition
def hml(x, lb=1.0, ub=3.0):
    """Returns vector of HIGH/MED/LOW for x given thresholds lb, ub, such that:
    x < lb --> LOW
    lb <= x < ub --> MED
    x >= ub --> HIGH"""

    def _hml(_x, lb, ub):
        rtnval = "LOW"
        if _x < lb:
            rtnval = "LOW"
        elif lb <= _x < ub:
            rtnval = "MED"
        else:
            rtnval = "HIGH"

        return rtnval

    hml = x.apply(_hml)
    hml = pd.Categorical(hml, categories=["LOW", "MED", "HIGH"], ordered=True)
    return hml


# %% Calculate betas for each sector relative to SPX for all market conditions
def calculate_betas(data, benchmark, label=None):
    all_results = []
    models = {}
    X = sm.add_constant(data[benchmark])  # Independent variable with constant

    for column in data.columns:
        if column != benchmark:
            # Perform the OLS regression
            model = sm.OLS(data[column], X).fit()
            models[column] = model
            # Gather results for this stock
            results = {
                "sector": column,
                "intercept": model.params["const"],
                "beta": model.params[benchmark],
                "nobs": int(model.nobs),
                "stderr": model.scale**0.5,
                "r2": model.rsquared,
                "r2adj": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
            }
            all_results.append(results)

    # Convert list of dictionaries to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.set_index("sector", inplace=True)
    if label is not None:
        results_df["label"] = f"{label.upper()}"

    return results_df, models
