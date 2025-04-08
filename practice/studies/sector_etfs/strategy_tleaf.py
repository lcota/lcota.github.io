import marimo

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    return


@app.cell
def _():
    from datetime import datetime, date
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sb
    # import modin.pandas as pd
    # import modin
    import pandas as pd
    from scipy.optimize import minimize 
    from utils import head, tail, lag, calculate_betas, hml
    from importlib import reload
    from collections import defaultdict

    import matplotlib.font_manager as fm
    return (
        calculate_betas,
        date,
        datetime,
        defaultdict,
        fm,
        head,
        hml,
        lag,
        minimize,
        mpl,
        np,
        pd,
        plt,
        reload,
        sb,
        tail,
    )


@app.cell
def _(fm):
    fm.fontManager.addfont("/Users/lcota/Library/Fonts/Raleway-Regular.ttf")
    fm.fontManager.addfont("/Users/lcota/Library/Fonts/Raleway-Light.ttf")
    return


@app.cell
def _(plt):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Raleway'
    plt.rcParams['font.style'] = "normal"
    # plt.rcParams['font.weight'] = 'Light'
    # plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 12
    # font.cursive
    # font.family
    # font.fantasy
    # font.monospace
    # font.sans-serif
    # font.serif
    # font.size
    # font.stretch
    # font.style
    # font.variant
    # font.weight
    # legend.fontsize
    # legend.title_fontsize
    # mathtext.fontset
    # pdf.fonttype
    # pdf.use14corefonts
    # pgf.rcfonts
    # ps.fonttype
    # svg.fonttype

    # plot_trading_rule(dfXcpi, dfrtn, xcol=xcol, rtn_col=rtn_col, 
    #                   weights=weights, save_plot=True, show_plot=True)

    # for k in plt.rcParams.keys():
    #     if "font" in k:
    #         print(k)
    return


@app.cell
def _():
    import strategies
    return (strategies,)


@app.cell
def _():
    from strategies import calc_excess_returns, factor_barplot, portfolio_performance, negative_sharpe_ratio, optimize_portfolio, calculate_factor_exposures, calculate_factor_exposures2, calc_strat_returns
    return (
        calc_excess_returns,
        calc_strat_returns,
        calculate_factor_exposures,
        calculate_factor_exposures2,
        factor_barplot,
        negative_sharpe_ratio,
        optimize_portfolio,
        portfolio_performance,
    )


@app.cell
def _():
    # dfcpi_fred = pd.read_parquet("workspace/cpi_fred_factors.pq").reset_index()/
    return


@app.cell
def _(date, pd):
    dfcpi = pd.read_parquet("workspace/cpi_turnleaf_factors.pq").reset_index()
    dfcpi['date'] = dfcpi['refdate'].apply(lambda x: date(x.year, x.month, 1))
    return (dfcpi,)


@app.cell
def _(pd):
    df_all = pd.read_parquet("workspace/dfreturns_with_tlfactors.pq")
    df_all['year'] = df_all['date'].apply(lambda x: pd.to_datetime(x).date().year)
    mask = (df_all['year'] >= 2013) & (df_all['year'] <= 2024)
    df_all = df_all[mask].copy()
    return df_all, mask


@app.cell
def _(df_all):
    df_all
    return


@app.cell
def _(df_all, dfcpi, head):
    factor_cols = ['date', 'refmonthyear', 'cpi.hml', 'pmi.hml', 'nmi.hml']
    _cpi_factors = ['date', 'refmonthyear', '3m_rising_yoy', '6m_rising_yoy', '9m_rising_yoy', '12m_rising_yoy', '3m1m_rising_yoy', '6m1m_rising_yoy', '9m1m_rising_yoy', '12m1m_rising_yoy']
    _rtn_cols = [c for c in df_all.columns if c.startswith('rtn') & c.endswith('M')]
    tickers = list(df_all['ticker'].unique())
    dfrtn = df_all.pivot(columns='ticker', index=['date', 'refmonthyear'], values=_rtn_cols)
    head(dfrtn)
    dfX = df_all[factor_cols].copy().drop_duplicates()
    dfX.set_index(['date', 'refmonthyear'], inplace=True)
    dfXcpi = dfcpi[_cpi_factors].copy().drop_duplicates()
    dfXcpi.set_index(['date', 'refmonthyear'], inplace=True)
    fwd_returns = {}
    df2 = dfrtn.stack(future_stack=True)
    df2.reset_index(inplace=True)
    for ticker in tickers:
        fwd_returns[ticker] = df2[df2['ticker'] == ticker].copy()
    del df2
    return df2, dfX, dfXcpi, dfrtn, factor_cols, fwd_returns, ticker, tickers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Trading Rule Return Functions""")
    return


@app.cell
def _():
    def get_weight(ticker, sigval, weights):
        if sigval is None:
            return 0
        return _weights[sigval][ticker]

    def zero():
        return 0
    return get_weight, zero


@app.cell
def _(defaultdict, get_weight, np, pd, plt, zero):
    def apply_trading_rule(dfX, dfrtn, xcol, rtn_col, weights):
        tickers = dfrtn[rtn_col].columns.values
        dfjoint = pd.merge(dfrtn[rtn_col].reset_index(), dfX[xcol].reset_index(), left_on='refmonthyear', right_on='refmonthyear', suffixes=('', '_y'), how='inner').dropna()
        dfjoint = dfjoint[dfjoint[xcol] != None]
        for ticker in tickers:
            wgtcol = f'wgt.{ticker}'
            wgtrtncol = f'wgtrtn.{ticker}'
            dfjoint[wgtcol] = dfjoint[xcol].apply(lambda x: get_weight(ticker, x, _weights))
            dfjoint[wgtrtncol] = dfjoint[wgtcol] * dfjoint[ticker]
        wgtreturns = dfjoint[[c for c in dfjoint.columns if c.startswith('wgtrtn')]].copy()
        wgtreturns['totrtn'] = np.cumprod(wgtreturns.agg(func='sum', axis='columns').apply(lambda x: 1 + x)) * 100
        wgtreturns = pd.DataFrame(wgtreturns['totrtn']).reset_index()
        wgtreturns['date'] = dfjoint['date'].copy()
        wgtreturns['refmonthyear'] = dfjoint['refmonthyear'].copy()
        wgtreturns = wgtreturns.sort_values('date')
        return wgtreturns

    def plot_trading_rule(dfX, dfrtn, xcol, rtn_col, weights, save_plot=True, show_plot=False, title=None, xlabel=None, ylabel=None):
        if _title is None:
            _title = f'totrtn.{xcol}.{rtn_col}'
        strat_return = apply_trading_rule(dfX, dfrtn, xcol, rtn_col, weights=_weights)
        mktweights = defaultdict(zero)
        spxlong = defaultdict(zero)
        spxlong['spx'] = 1
        mktweights[True] = spxlong
        mktweights[False] = spxlong
        mkt_return = apply_trading_rule(dfX, dfrtn, xcol, rtn_col, weights=mktweights)
        if not show_plot:
            plt.ioff()
        plt.figure(figsize=(10, 4))
        plt.plot(strat_return['date'], strat_return['totrtn'], label='Long/Short')
        plt.plot(mkt_return['date'], mkt_return['totrtn'], label='SPX')
        plt.title(_title)
        plt.legend()
        plt.grid(ls='--', alpha=0.5)
        if _xlabel is not None:
            plt.xlabel(_xlabel)
        if _ylabel is not None:
            plt.ylabel(_ylabel)
        if save_plot:
            figname = f'img/trading_rule_plots/{_title}.png'
            plt.savefig(figname, dpi=200)
        if show_plot:
            plt.show()
        plt.close()
        plt.ion()
    return apply_trading_rule, plot_trading_rule


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Strategy 1: Long/Short top 2 tickers 
        Rising 3Mth inflation ~ 6Mth Return:  
            Long Tech  ```s5tech```  
            Long Energy  ```s5enrs```  
            Short Cons Staples  ```s5cons```  
            Short Healthcare  ```s5hlth```  

        Falling 3Mth inflation ~ 6Mth Return:  
            Long Tech  ```s5tech```  
            Long Materials  ```s5matr```  
            Short Cons Staples  ```s5cons```  
            Short Financials  ```s5finl```
        """
    )
    return


@app.cell
def _(apply_trading_rule, defaultdict, dfXcpi, dfrtn, plot_trading_rule, zero):
    _rising_weights = defaultdict(zero)
    _falling_weights = defaultdict(zero)
    _weights = defaultdict(zero)
    _rising_weights['s5cond'] = 1
    _rising_weights['s5hlth'] = 1
    _rising_weights['s5indu'] = -1
    _rising_weights['s5finl'] = -1
    _falling_weights['s5tech'] = 1
    _falling_weights['s5enrs'] = 1
    _falling_weights['s5indu'] = -1
    _falling_weights['s5hlth'] = -1
    _weights[True] = _rising_weights
    _weights[False] = _falling_weights
    xcol = '3m_rising_yoy'
    rtn_col = 'rtn_1M'
    _title = '{} Fwd Returns ~ 3M Rising CPI YoY\nTurnleaf Forecast Data'.format(rtn_col.replace('rtn_', ''))
    _xlabel = 'Date'
    _ylabel = 'Total Return'
    strat_rtn = apply_trading_rule(dfXcpi, dfrtn, xcol, rtn_col, _weights)
    plot_trading_rule(dfXcpi, dfrtn, xcol=xcol, rtn_col=rtn_col, weights=_weights, save_plot=True, show_plot=True, title=_title, xlabel=_xlabel, ylabel=_ylabel)
    return rtn_col, strat_rtn, xcol


@app.cell
def _(dfXcpi, dfrtn, pd, rtn_col, xcol):
    pd.merge(dfrtn[rtn_col].reset_index(), 
             dfXcpi[xcol].reset_index(), 
             left_on='refmonthyear', 
             right_on='refmonthyear', 
             suffixes=('', '_y'),
             how='inner')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Strategy 2: Long/Short top 2 tickers 
        Rising 3Mth inflation ~ 3Mth Return:  
            Long Tech  ```s5tech```  
            Long Fins  ```s5finl```  
            Short Energy  ```s5enrs```  
            Short Industrials  ```s5indu```  

        Falling 3Mth inflation ~ 3Mth Return:  
            Long Energy  ```s5enrs```  
            Long Tech  ```s5tech```  
            Short Fins  ```s5finl```  
            Short Cons Staples  ```s5cons```
        """
    )
    return


@app.cell
def _(defaultdict, dfXcpi, dfrtn, plot_trading_rule, zero):
    _rising_weights = defaultdict(zero)
    _falling_weights = defaultdict(zero)
    _weights = defaultdict(zero)
    _rising_weights['s5hlth'] = 1
    _rising_weights['s5matr'] = 1
    _rising_weights['s5indu'] = -1
    _rising_weights['s5enrs'] = -1
    _falling_weights['s5tech'] = 1
    _falling_weights['s5finl'] = 1
    _falling_weights['s5hlth'] = -1
    _falling_weights['s5indu'] = -1
    _weights[True] = _rising_weights
    _weights[False] = _falling_weights
    rtn_col_1 = 'rtn_2M'
    _title = '{} Fwd Returns ~ 3M Rising CPI YoY\nTurnleaf Forecast Data'.format(rtn_col_1.replace('rtn_', ''))
    _xlabel = 'Date'
    _ylabel = 'Total Return'
    xcol_1 = '3m_rising_yoy'
    plot_trading_rule(dfXcpi, dfrtn, xcol=xcol_1, rtn_col=rtn_col_1, weights=_weights, save_plot=True, show_plot=True, title=_title, xlabel=_xlabel, ylabel=_ylabel)
    return rtn_col_1, xcol_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Strategy 3""")
    return


@app.cell
def _(defaultdict, dfXcpi, dfrtn, plot_trading_rule, zero):
    rtn_col_2 = 'rtn_1M'
    xcol_2 = '6m_rising_yoy'
    _rising_weights = defaultdict(zero)
    _falling_weights = defaultdict(zero)
    _weights = defaultdict(zero)
    _rising_weights['s5tech'] = 1
    _rising_weights['s5hlth'] = 1
    _rising_weights['s5indu'] = -1
    _rising_weights['s5enrs'] = -1
    _falling_weights['s5tech'] = 1
    _falling_weights['s5enrs'] = 1
    _falling_weights['s5hlth'] = -1
    _falling_weights['s5finl'] = -1
    _weights[True] = _rising_weights
    _weights[False] = _falling_weights
    _title = '{} Fwd Returns ~ {} Rising CPI YoY\nTurnleaf Forecast Data'.format(rtn_col_2.replace('rtn_', ''), xcol_2.replace('_rising_yoy', ''))
    _xlabel = 'Date'
    _ylabel = 'Total Return'
    plot_trading_rule(dfXcpi, dfrtn, xcol=xcol_2, rtn_col=rtn_col_2, weights=_weights, save_plot=True, show_plot=True, title=_title, xlabel=_xlabel, ylabel=_ylabel)
    return rtn_col_2, xcol_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Strategy 4""")
    return


@app.cell
def _(defaultdict, dfXcpi, dfrtn, plot_trading_rule, zero):
    rtn_col_3 = 'rtn_2M'
    xcol_3 = '6m_rising_yoy'
    _rising_weights = defaultdict(zero)
    _falling_weights = defaultdict(zero)
    _weights = defaultdict(zero)
    _rising_weights['s5tech'] = 1
    _rising_weights['s5finl'] = 1
    _rising_weights['s5indu'] = -1
    _rising_weights['s5enrs'] = -1
    _falling_weights['s5tech'] = 1
    _falling_weights['s5cond'] = 1
    _falling_weights['s5hlth'] = -1
    _falling_weights['s5cons'] = -1
    _weights[True] = _rising_weights
    _weights[False] = _falling_weights
    _title = '{} Fwd Returns ~ {} Rising CPI YoY\nTurnleaf Forecast Data'.format(rtn_col_3.replace('rtn_', ''), xcol_3.replace('_rising_yoy', ''))
    _xlabel = 'Date'
    _ylabel = 'Total Return'
    plot_trading_rule(dfXcpi, dfrtn, xcol=xcol_3, rtn_col=rtn_col_3, weights=_weights, save_plot=True, show_plot=True, title=_title, xlabel=_xlabel, ylabel=_ylabel)
    return rtn_col_3, xcol_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Strategy 5""")
    return


@app.cell
def _(defaultdict, dfXcpi, dfrtn, plot_trading_rule, zero):
    rtn_col_4 = 'rtn_6M'
    xcol_4 = '6m_rising_yoy'
    _rising_weights = defaultdict(zero)
    _falling_weights = defaultdict(zero)
    _weights = defaultdict(zero)
    _rising_weights['s5tech'] = 1
    _rising_weights['s5hlth'] = 1
    _rising_weights['s5finl'] = -1
    _rising_weights['s5indu'] = -1
    _falling_weights['s5tech'] = 1
    _falling_weights['s5cond'] = 1
    _falling_weights['s5hlth'] = -1
    _falling_weights['s5cons'] = -1
    _weights[True] = _rising_weights
    _weights[False] = _falling_weights
    _title = '{} Fwd Returns ~ {} Rising CPI YoY\nTurnleaf Forecast Data'.format(rtn_col_4.replace('rtn_', ''), xcol_4.replace('_rising_yoy', ''))
    _xlabel = 'Date'
    _ylabel = 'Total Return'
    plot_trading_rule(dfXcpi, dfrtn, xcol=xcol_4, rtn_col=rtn_col_4, weights=_weights, save_plot=True, show_plot=True, title=_title, xlabel=_xlabel, ylabel=_ylabel)
    return rtn_col_4, xcol_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# $\Delta cpi.yoy\$ ~ Mean Return x sector""")
    return


@app.cell
def _(dfX, dfXcpi, pd):
    dfX_1 = pd.merge(left=dfX.reset_index(), right=dfXcpi.reset_index(), left_on='refmonthyear', right_on='refmonthyear', suffixes=(None, '_y'))
    dfX_1.drop(columns=[x for x in dfX_1.columns if x.endswith('_y')], axis=1, inplace=True)
    dfX_1.set_index(['date', 'refmonthyear'], inplace=True)
    return (dfX_1,)


@app.cell
def _(dfrtn, head):
    dfrtn2 = dfrtn.stack(future_stack=True).reset_index()
    head(dfrtn2)
    return (dfrtn2,)


@app.cell
def _(dfXcpi, dfrtn2, pd):
    rtn_col_5 = 'rtn_1M'
    xcol_5 = '3M_rising_yoy'
    dfrtn_cpi2 = pd.merge(dfrtn2, dfXcpi.reset_index(), left_on='refmonthyear', right_on='refmonthyear', suffixes=('', '_y'), how='inner').drop('date_y', axis='columns')
    return dfrtn_cpi2, rtn_col_5, xcol_5


@app.cell
def _():
    _cpi_factors = ['3M_rising_yoy', '6M_rising_yoy', '9M_rising_yoy', '12M_rising_yoy']
    _rtn_cols = [f'rtn_{i + 1}M' for i in range(12)]
    return


@app.cell
def _(dfrtn_cpi2):
    dfrtn_cpi2[dfrtn_cpi2.ticker == 'spx'].to_clipboard()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Changes since last call - updated signal to work on relative changes in YoY CPI instead of absolute -- now roughly 50/50 split between rising & falling regimes across CPI signal horizons.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Strategy 3: Long/Short top 2 tickers with CPI Signal v2
        Rising 3Mth inflation ~ 3Mth Return:  
            Long Tech  ```s5tech```  
            Long Healthcare  ```s5hlth```  
            Short Energy  ```s5enrs```  
            Short Industrials  ```s5indu```  

        Falling 3Mth inflation ~ 3Mth Return:  
            Long Tech  ```s5tech```  
            Long Fins  ```s5finl```  
            Short Healthcare  ```s5hlth```  
            Short Cons Staples  ```s5cons```
        """
    )
    return


@app.cell
def _():
    xcol_6 = '3m1m_rising_yoy'
    rtn_col_6 = 'rtn_3M'
    _title = 'Excess {} Fwd Sector Returns ~ 3M Rising CPI YoY\nTurnleaf Forecast Data'.format(rtn_col_6.replace('rtn_', ''))
    _ylabel = 'Excess {} Fwd Return'.format(rtn_col_6.replace('rtn_', ''))
    _xlabel = 'Sector Ticker'
    return rtn_col_6, xcol_6


@app.cell
def _(defaultdict, dfX_1, dfrtn, plot_trading_rule, zero):
    _rising_weights = defaultdict(zero)
    _falling_weights = defaultdict(zero)
    _weights = defaultdict(zero)
    _rising_weights['s5enrs'] = 1
    _rising_weights['s5finl'] = 1
    _rising_weights['s5hlth'] = -1
    _rising_weights['s5indu'] = -1
    _falling_weights['s5hlth'] = 1
    _falling_weights['s5enrs'] = 1
    _falling_weights['s5cond'] = -1
    _falling_weights['s5indu'] = -1
    _weights[True] = _rising_weights
    _weights[False] = _falling_weights
    xcol_7 = '3m_rising_yoy'
    rtn_col_7 = 'rtn_1M'
    _title = '{} Fwd Returns ~ 3M Rising CPI YoY\nTurnleaf Forecast Data'.format(rtn_col_7.replace('rtn_', ''))
    _xlabel = 'Date'
    _ylabel = 'Total Return'
    plot_trading_rule(dfX_1, dfrtn, xcol=xcol_7, rtn_col=rtn_col_7, weights=_weights, save_plot=True, show_plot=True, title=_title, xlabel=_xlabel, ylabel=_ylabel)
    return rtn_col_7, xcol_7


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Strategy 4: Long/Short top 2 tickers with CPI Signal v2, 3M Rtn ~ 6M CPI Rising
        Rising 3Mth Return ~ 6Mth Inflation:  
            Long Tech  ```s5tech```  
            Long Healthcare  ```s5hlth```  
            Short Cons Disc  ```s5cond```  
            Short Industrials  ```s5indu```  

        Falling 3Mth Return ~ 6Mth Inflation:  
            Long Tech  ```s5tech```  
            Long Cons Disc  ```s5cond```  
            Short Healthcare  ```s5hlth```  
            Short Cons Staples  ```s5cons```
        """
    )
    return


@app.cell
def _(dfrtn_cpi2, factor_barplot):
    xcol_8 = '6m_rising_yoy'
    rtn_col_8 = 'rtn_3M'
    _title = '3M Fwd Returns ~ 6M Rising CPI YoY\nTurnleaf Forecast Data'
    _ylabel = 'Excess 3M Fwd Return'
    _xlabel = 'Sector Ticker'
    factor_barplot(dfrtn_cpi2, xcol_8, rtn_col_8, show_plot=True, save_plot=True, figsize=(10, 4), ylabel=_ylabel, xlabel=_xlabel, title=_title)
    return rtn_col_8, xcol_8


@app.cell
def _(defaultdict, dfX_1, dfrtn, plot_trading_rule, rtn_col_8, xcol_8, zero):
    _rising_weights = defaultdict(zero)
    _falling_weights = defaultdict(zero)
    _weights = defaultdict(zero)
    _rising_weights['s5enrs'] = 1
    _rising_weights['s5finl'] = 1
    _rising_weights['s5cons'] = -1
    _rising_weights['s5indu'] = -1
    _falling_weights['s5matr'] = 1
    _falling_weights['s5hlth'] = 1
    _falling_weights['s5cond'] = -1
    _falling_weights['s5finl'] = -1
    _weights[True] = _rising_weights
    _weights[False] = _falling_weights
    _title = '3M Fwd Returns ~ 6M Rising CPI YoY\nTurnleaf Forecast Data'
    _xlabel = 'Date'
    _ylabel = 'Total Return'
    plot_trading_rule(dfX_1, dfrtn, xcol=xcol_8, rtn_col=rtn_col_8, weights=_weights, save_plot=True, show_plot=True, title=_title, xlabel=_xlabel, ylabel=_ylabel)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Strategy 5: Long/Short top 2 tickers with CPI Signal v2, 6M Rtn ~ 6M CPI Rising
        Rising 6Mth Return ~ 6Mth Inflation:  
            Long Tech  ```s5tech```  
            Long Energy  ```s5enrs```  
            Short Fins  ```s5finl```  
            Short Cons Disc  ```s5cond```  

        Falling 6Mth Return ~ 6Mth Inflation:  
            Long Tech  ```s5tech```  
            Long Fins  ```s5finl```  
            Short Healthcare  ```s5hlth```  
            Short Cons Staples  ```s5cons```
        """
    )
    return


@app.cell
def _(dfrtn_cpi2, factor_barplot, plt):
    xcol_9 = '6m_rising_yoy'
    rtn_col_9 = 'rtn_6M'
    _title = '6M Fwd Returns ~ 6M Rising CPI YoY'
    _ylabel = 'Excess 6M Fwd Return'
    _xlabel = 'Sector Ticker'
    plt.rcParams['figure.figsize'] = (6, 3)
    factor_barplot(dfrtn_cpi2, xcol_9, rtn_col_9, show_plot=True, save_plot=False, figsize=(6, 3), ylabel=_ylabel, xlabel=_xlabel)
    return rtn_col_9, xcol_9


@app.cell
def _(defaultdict, dfX_1, dfrtn, plot_trading_rule, rtn_col_9, xcol_9, zero):
    _rising_weights = defaultdict(zero)
    _falling_weights = defaultdict(zero)
    _weights = defaultdict(zero)
    _rising_weights['s5tech'] = 1
    _rising_weights['s5enrs'] = 1
    _rising_weights['s5finl'] = -1
    _rising_weights['s5cond'] = -1
    _falling_weights['s5tech'] = 1
    _falling_weights['s5finl'] = 1
    _falling_weights['s5hlth'] = -1
    _falling_weights['s5cons'] = -1
    _weights[True] = _rising_weights
    _weights[False] = _falling_weights
    plot_trading_rule(dfX_1, dfrtn, xcol=xcol_9, rtn_col=rtn_col_9, weights=_weights, save_plot=True, show_plot=True)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
