import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as ts 
import datetime
import QuantLib as ql
import matplotlib.pyplot as plt
import seaborn as sb
import re


from pandas import Series, DataFrame, melt
from datatools import *

# nyse = mcal.get_calendar('NYSE')
# def get_term_structure()

#%% Data manipulation

calendars = {
    'USA': ql.UnitedStates(ql.UnitedStates.NYSE)
}

country_code_map = {
    'Argentina': 'ARG',
    'Australia': 'AUS',
    'Brazil': 'BRL',
    'Canada': 'CAN',
    'Chile' : 'CHL',
    'China': 'CHI',
    'Colombia': 'COL',
    'Czech Republic': 'CZE',
    'Eurozone': 'EUR',
    'France' : 'FRA',
    'Germany': 'GER',
    'Hungary': 'HUN',
    
    'India': 'IND',
    'Indonesia': 'IDN',
    'Israel' : 'ISR',
    'Italy' : 'ITA',
    'Japan' : 'JPN',
    'Mexico': 'MEX',
    'Netherlands': 'NED',
    'Peru': 'PER',
    'Poland':'POL',
    'Romania': 'ROU',
    'South Africa': 'RSA', # Republic of South Africa
    'South Korea': 'KOR',
    'Spain': 'ESP',
    'Sweden': 'SWE',
    'Switzerland': 'SUI',
    'Thailand': 'THA',
    'Turkey': 'TUR',
    'United Kingdom': 'GBR',
    'United States': "USA"
}


def replace_country_names(df, use_copy=False):
    if use_copy:
        _df = df.copy()
    else:
        _df = df
    colnames = _df.columns

    newcols = []
    for cname in colnames:
        for k, v in country_code_map.items():
            if k in cname:
                newcol = cname.replace(f"{k} ", f"{v}.")
                newcols.append(newcol)
                break
            # else:
            #     newcols.append(cname)
            #     # print(cname)


    if len(newcols) == len(colnames):
        _df.columns = newcols
    return _df


def simplify_cols(df, use_copy=False):
    if use_copy:
        _df = df.copy()
    else:
        _df = df
        
    colnames = _df.columns
    newcols = []
    for cname in colnames:
        newcol = ''
        if "Turnleaf Backtest Forecast" in cname:
            newcol = cname.replace("Turnleaf Backtest Forecast ", "tlbt.")

        elif "Turnleaf Consensus Backtest Forecast" in cname:
            newcol = cname.replace("Turnleaf Consensus Backtest Forecast ", "tlbt.cons.")

        else:
            # print(cname)
            newcol = cname

        newcol = newcol.replace('CPI YoY NSA ', 'cpi.yoy.nsa.')
        newcol = newcol.replace("CPI MoM NSA ", "cpi.mom.nsa.")
        newcol = newcol.replace('CPI YoY SA ', 'cpi.yoy.sa.')
        newcol = newcol.replace("CPI MoM SA ", "cpi.mom.sa.")
        newcol = newcol.replace('ISM Manufacturing Index SA ', 'ism.man.')
        newcol = newcol.replace('ISM Non Manufacturing Index SA ', 'ism.nonman.')
        newcol = newcol.replace("Realised ", "realised.")
        newcol = newcol.replace("Prediction Horizon for ", "date.")
        newcol = newcol.replace("Forecast Date", "pubdate")
        newcol = newcol.replace("Reference Date", "refdate")
        newcol = newcol.lower()

        newcols.append(newcol)
        
    if len(newcols) == len(colnames):
        _df.columns = newcols
    
    return _df
    

def filter_cols(df, str_filter='usa'):
    cols2keep = [col for col in df.columns if re.match(str_filter, col)]
    
    if len(cols2keep) > 0:
        return df[cols2keep]
    
    return df


def qldate_to_date(qldate):
    return datetime.date(qldate.Year, qldate.month(), qldate.day())
                         
def date_to_qldate(date):
    return ql.Date(date.day, date.month, date.year)

def adj_bus_days(date, calendar, mod_backward=True):
    qldate = date_to_qldate(date)
    cdr = calendars[calendar]
    if cdr.isBusinessDay(qldate):
        return qldate.to_date()
    else:        
        if mod_backward:
            ndays = -1
        else:
            ndays = 1
            
        while not cdr.isBusinessDay(qldate):
            qldate = cdr.advance(qldate, ndays, ql.Days)
            
        return qldate.to_date()
    
def adj_date(x, ndays=3, force=False, mod_backward=False, country='USA'):
    if x is None or x==pd.NaT:
        print(x, "x is NaT or None")
        return pd.NaT
        # x = datetime.date(2023, 12, 12)
    dt = pd.to_datetime(x).date()
    if x is pd.NaT:
        return pd.NaT
    try:
        if force or dt < datetime.date(2020, 3, 20): #cutoff date for adjustment is all dates prior to March 2020
            dt += pd.tseries.offsets.BusinessDay(ndays)
            dt = pd.to_datetime(dt).date()

        dt = adj_bus_days(dt, country, mod_backward=mod_backward)
    except:
        print(dt)
    return dt

def add_pubdate(df):
    df['pubdate'] = pd.to_datetime(df['pubdate'], format='%Y%m%d')
    return df


###################################################################################
#
#%% Stats utility functions

def adf_test(tseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = ts.adfuller(tseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['TStat','p-value','#Lags Used',
                                             'NObs'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)
    return dfoutput

def calc_ccf(x, returns, labels, nmonths=12):
    netfs = len(labels)
    nmonths = nmonths # months of lag to examine 
    
    # sm.tsa.ccf, sm.tsa.ccovf, sm.tsa.pacf, sm.tsa.pacf_ols
    # x1 = df_cpi_diffs['cpi.mom.diff']
    corr_mat = np.zeros(shape=(netfs, nmonths))
    i = 0
    for sym in returns.columns:
        rtn = returns[sym]
        cc = ts.ccf(x, rtn)[0:nmonths]
        corr_mat[i] = np.round(cc * 100., 2)
        i += 1

    cols = ["{}M".format(i) for i in range(1, nmonths+1)]
    df = pd.DataFrame(data=corr_mat, columns=cols, index=labels)
    
    return df


def calc_univariate_betas(x, returns, refdata, lag=0):
    x1 = x.shift(lag).values
    # X = np.vstack(zip(x1, x2))
    X = sm.add_constant(x1)
    ols_models = {}
    # ols_summaries = {}
    results = {'sym': [], 'assetclass': [],  'const': [], 'x1': [], 'r2': []}

    for sym in returns.columns:
        assetclass = refdata.loc[sym]['assetclass']        
        y = returns[sym]
        # _df = pd.DataFrame(y)
        mdl = sm.OLS(y, X, missing='drop').fit()
        # print(mdl.summary())
        ols_models[sym] = mdl
        results['sym'].append(sym)        
        results['assetclass'].append(assetclass)
        results['const'].append(mdl.params['const'])
        results['x1'].append(mdl.params['x1'])
        # results['b2'].append(mdl.params['x2'])
        results['r2'].append(mdl.rsquared)
        # results['r2adj'].append(mdl.rsquared_adj)

    df = pd.DataFrame(results).sort_values(by=['assetclass'], ascending=False)
    return df


def mean_ols_stats(dfX, returns, refdata, nlags=13, xcol='cpi.mom.diff'):
    row_labels = [f"{i}M" for i in range(nlags)]
    row_labels = pd.Categorical(row_labels, categories=row_labels, ordered=True)
    
    avg_betas = []
    avg_r2 = []

    for i in range(nlags):
        _df = calc_univariate_betas(dfX[xcol], 
                                    returns, 
                                    refdata, 
                                    lag=i)
        avg_betas.append(pd.DataFrame(_df.groupby('assetclass')['x1'].mean()).T)
        avg_r2.append(pd.DataFrame(_df.groupby('assetclass')['r2'].mean()).T)
        
    dfbetas = pd.concat(avg_betas)
    dfbetas.index = row_labels # range(len(avg_betas))
    dfbetas.reset_index(inplace=True)
    dfbetas.columns = ['lag', 'bcmdy', 'bdebt', 'beqty']
    dfbetas['endog'] = xcol
    
    dfr2 = pd.concat(avg_r2)
    dfr2.index = row_labels # range(len(avg_r2))
    dfr2.reset_index(inplace=True)
    dfr2.columns = ['lag', 'bcmdy', 'bdebt', 'beqty']
    dfr2['endog'] = xcol
    
    dfbetas = pd.melt(dfbetas, 
                  id_vars = ['endog', 'lag'], 
                  value_vars=['bcmdy', 'bdebt', 'beqty'], 
                  var_name='asset_class',
                  value_name='beta')

    dfr2 = pd.melt(dfr2,
                id_vars = ['endog', 'lag'], 
                value_vars=['bcmdy', 'bdebt', 'beqty'], 
                var_name='asset_class',
                value_name='r2')
    
    
    return dfbetas, dfr2


def plot_ols_stats(dfX, returns, refdata, nlags=13, xcol='cpi.mom.diff', title="Monthly Return"):
    dfbetas, dfr2 = mean_ols_stats(dfX, returns, refdata, nlags, xcol=xcol)

    title_beta = f"Beta\n{title} ~ {xcol}"
    title_r2 = f"R2\n{title} ~ {xcol}"

    plt.subplot(121);
    sb.lineplot(data=dfbetas, x='lag', y='beta', 
                hue='asset_class', lw=2);
    # plt.title("Monthly Return ~ CPI Diff Beta \n CPI MoM");
    plt.title(title_beta)

    plt.subplot(122);
    sb.lineplot(data=dfr2, x='lag', y='r2', 
                hue='asset_class', lw=2);
    # plt.title("Monthly Return ~ CPI Diff R2 \n CPI MoM");
    plt.title(title_r2)
    

#########################################################################################
# Backtesting Support Functions
#


def get_trade_dates(df):
    dates = pd.to_datetime(df.index.values)
    dates = pd.Series(dates).apply(lambda x: pd.to_datetime(x))
    df_trades = pd.DataFrame(dates, columns=['index'])
    df_trades['entry'] = dates
    df_trades['trd.exit1w'] = df_trades['entry'].apply(lambda x: adj_date(x, 5, force=True, mod_backward=True))
    df_trades['trd.exit2w'] = df_trades['entry'].apply(lambda x: adj_date(x, 10, force=True, mod_backward=True))
    df_trades['trd.exit3w'] = df_trades['entry'].apply(lambda x: adj_date(x, 15, force=True, mod_backward=True))
    df_trades['trd.exit1m'] = df_trades['entry'].apply(lambda x: adj_date(x, 20, force=True, mod_backward=True))
        
    # df_trades['trd.exit1m'] = (df_trades['entry']
    #                             .shift(-1)
    #                             .apply(lambda x: adj_date(x, -1, force=True, mod_backward=True)))
        
    df_trades.set_index('index', inplace=True)
    
    return df_trades

def get_trade_price(prices, ticker, date, col='adjusted'):
    pass

def get_trade_prices(prices, tickers, dates, col='adjusted'):
    df = prices.pivot(columns='ticker', values=col)[tickers]
    return df.loc[dates]
    

def calc_returns(dfdates, close_prices, entry_dates, exit_dates):
    buys = entry_dates.merge(right=close_prices, 
                              how='left',
                              left_index=True,
                              right_index=True,
                              suffixes=(None, None)).values

    sells = exit_dates.merge(right=close_prices,
                              how='left',
                              left_index=True,
                              right_index=True,
                              suffixes=(None, None)).values

    df_trade_returns = pd.DataFrame(sells / buys - 1.0, 
                                    index=dfdates['pubdate'], 
                                    columns=close_prices.columns)

    return df_trade_returns

#########################################################################################
# Plotting Utility Functions
#
from datatools import roll_quantile
def quadrant_plot(dfX, dfY, xcol, ycol, 
                  title=None, 
                  xcol_name=None, 
                  quantile=.5):
    
    if xcol_name is None:
        colname = f"trailing.{xcol}.q{int(quantile*100)}"
    else:
        colname = xcol_name
        
    if title is None:        
        title = f"{ycol.upper()} vs {colname}"
    
    
    qvalues = roll_quantile(dfX[xcol], colname=colname, quantile=quantile)        
    X = dfX[xcol].to_frame()
    X[colname] = qvalues[colname]
    # X['above_qval'] = X[srccol] > X[colname]
    
    dfrtn = dfY[ycol].resample("1M").agg("ohlc")['close'] / dfY[ycol].resample("1M").agg("ohlc")['open'] - 1
    dfrtn.name = f'{ycol}'
    # data = pd.merge_asof(dfrtn, dfX, left_index=True, right_on='ref_date', direction='backward')
    # data = pd.merge_asof(dfrtn, X, left_index=True, right_on='ref_date', direction='backward')
    data = pd.merge_asof(dfrtn, X, left_index=True, right_index=True, direction='backward')
    data['posreturn'] = data[f'{ycol}'] > 0
    data['above_quantile'] = data[xcol] > data[colname]
    
    palette = {True: "tab:blue", False: "tab:orange"}

    sb.scatterplot(data, x=colname, y=f'{ycol}', 
                   hue='posreturn',                    
                   alpha=1, 
                   palette=palette,
                   # palette='vlag_r', 
                   legend=True)
    leg = plt.legend(labels=['Pos. Return', 'Neg. Return'], fontsize=8, loc='upper center', ncols=2)
    leg.legendHandles[0].set_color('tab:blue')
    leg.legendHandles[1].set_color('tab:orange')
    plt.axvline(X[colname].quantile(quantile), color='#9b9b9b', linestyle='--', alpha=.75, lw=2)
    plt.axhline(0, color='#9b9b9b', linestyle='--', alpha=.75, lw=2)
    plt.xlabel(colname, fontsize=8)
    plt.ylabel(ycol.upper(), fontsize=8)
    plt.title(title, fontsize=9)





