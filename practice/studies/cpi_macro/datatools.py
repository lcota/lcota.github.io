import numpy as np
import pandas as pd
# import datatable as dt

def roll_quantile(series, colname, quantile=.5, window=12, minperiods=6):
    """
    Calculate rolling quantile of a column of a dataframe.
    """
    # colname = f"{colprefix}.q{int(quantile*100)}"
    qvalues = series.rolling(window=window, min_periods=minperiods).quantile(quantile)
    qvalues = pd.DataFrame(qvalues)
    qvalues.columns = [colname]
    return qvalues

def head(df, n=5):
    return df[0:n]

def tail(df, n=5):
    return df[-n:]

def lag(df, n=1):
    '''Assumes df is sorted in ascending order by date'''
    return df.shift(n)