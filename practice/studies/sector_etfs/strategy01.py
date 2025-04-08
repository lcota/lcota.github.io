import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Imports Setup""")
    return


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
    from datetime import datetime
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sb

    # import modin.pandas as pd
    # import modin
    import pandas as pd
    from scipy.optimize import minimize
    from utils import head, tail, hml, calculate_betas, lag
    from importlib import reload
    from collections import defaultdict

    import matplotlib.font_manager as fm
    return (
        calculate_betas,
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
def _(plt):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Raleway"
    plt.rcParams["font.style"] = "normal"
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
    from strategies import (
        calc_excess_returns,
        factor_barplot,
        portfolio_performance,
        negative_sharpe_ratio,
        optimize_portfolio,
        calculate_factor_exposures,
        calculate_factor_exposures2,
        calc_strat_returns,
    )
    return (
        calc_excess_returns,
        calc_strat_returns,
        calculate_factor_exposures,
        calculate_factor_exposures2,
        factor_barplot,
        negative_sharpe_ratio,
        optimize_portfolio,
        portfolio_performance,
        strategies,
    )


@app.cell
def _(pd):
    dfcpi_fred = pd.read_parquet("workspace/cpi_fred_factors.pq").reset_index()
    return (dfcpi_fred,)


@app.cell
def _(dfcpi_fred, pd, pvt_rtn):
    df_all = pd.read_parquet("workspace/dfreturns_with_factors.pq")


    factor_cols = [
        "date",
        "refmonthyear",
        "cpi.hml",
        "pmi.hml",
        "nmi.hml",
        "3M_rising",
        "6M_rising",
        "9M_rising",
        "12M_rising",
    ]

    cpi_factors = [
        "date",
        "refmonthyear",
        "3M_rising",
        "6M_rising",
        "9M_rising",
        "12M_rising",
        "3M_rising_yoy",
        "6M_rising_yoy",
        "9M_rising_yoy",
        "12M_rising_yoy",
    ]

    rtn_cols = [c for c in df_all.columns if c.startswith("rtn") & c.endswith("M")]
    rtn_cols = ["rtn"] + rtn_cols

    tickers = list(df_all["ticker"].unique())

    # Need to unstack returns per sector to split factor columns
    dfrtn = df_all.pivot(
        columns="ticker", index=["date", "refmonthyear"], values=rtn_cols
    )
    # head(dfrtn)

    dfX = df_all[factor_cols].copy().drop_duplicates()
    dfX.set_index(["date", "refmonthyear"], inplace=True)


    dfXcpi = dfcpi_fred[cpi_factors].copy().drop_duplicates()
    dfXcpi.set_index(["date", "refmonthyear"], inplace=True)

    fwd_returns = {}

    df2 = dfrtn.stack(future_stack=True)
    df2.reset_index(inplace=True)

    for ticker in tickers:
        fwd_returns[ticker] = df2[df2["ticker"] == ticker].copy()

    del df2

    keep_cols = ["refmonthyear", "ticker", "date"] + rtn_cols
    dfrtn_simple = df_all[keep_cols].copy()
    # dfrtn_simple = dfrtn_simple.set_index([""])
    del keep_cols

    dfreturns = {}
    for rc in rtn_cols:
        dfreturns[rc] = pvt_rtn(df=dfrtn_simple, rtncol=rc)
    return (
        cpi_factors,
        df2,
        dfX,
        dfXcpi,
        df_all,
        dfreturns,
        dfrtn,
        dfrtn_simple,
        factor_cols,
        fwd_returns,
        keep_cols,
        rc,
        rtn_cols,
        ticker,
        tickers,
    )


@app.cell
def _():
    def get_weight(ticker, sigval, weights):
        if sigval is None:
            return 0
        return weights[sigval][ticker]


    def zero():
        return 0
    return get_weight, zero


@app.cell
def _(defaultdict, get_weight, np, pd, plt, zero):
    def pvt_rtn(df, rtncol):
        return df.pivot(
            columns="ticker", index=["date", "refmonthyear"], values=rtncol
        )


    def apply_trading_rule(dfX, dfrtn, xcol, rtn_col, weights):
        tickers = dfrtn[rtn_col].columns.values

        # strat_code = f"totrtn.{xcol}.{rtn_col}"

        dfjoint = pd.merge(
            dfrtn[rtn_col].reset_index(),
            dfX[xcol].reset_index(),
            left_on="refmonthyear",
            right_on="refmonthyear",
            suffixes=("", "_y"),
            how="inner",
        ).dropna()
        dfjoint = dfjoint[dfjoint[xcol] != None]

        for ticker in tickers:
            wgtcol = f"wgt.{ticker}"
            wgtrtncol = f"wgtrtn.{ticker}"
            dfjoint[wgtcol] = dfjoint[xcol].apply(
                lambda x: get_weight(ticker, x, weights)
            )
            dfjoint[wgtrtncol] = dfjoint[wgtcol] * dfjoint[ticker]

        wgtreturns = dfjoint[
            [c for c in dfjoint.columns if c.startswith("wgtrtn")]
        ].copy()
        # wgtreturns[strat_code] = np.cumprod(wgtreturns.agg(func='sum', axis='columns').apply(lambda x: 1+x)) * 100
        wgtreturns["totrtn"] = (
            np.cumprod(
                wgtreturns.agg(func="sum", axis="columns").apply(lambda x: 1 + x)
            )
            * 100
        )
        wgtreturns = pd.DataFrame(wgtreturns["totrtn"]).reset_index()
        wgtreturns["date"] = dfjoint["date"].copy()
        wgtreturns["refmonthyear"] = dfjoint["refmonthyear"].copy()
        wgtreturns = wgtreturns.sort_values("date")
        # wgtreturns.set_index('date', inplace=True)
        return wgtreturns


    def plot_trading_rule(
        dfX,
        dfrtn,
        xcol,
        rtn_col,
        weights,
        save_plot=True,
        show_plot=False,
        title=None,
        xlabel=None,
        ylabel=None,
    ):
        if title is None:
            title = f"totrtn.{xcol}.{rtn_col}"
        strat_return = apply_trading_rule(
            dfX, dfrtn, xcol, rtn_col, weights=weights
        )

        mktweights = defaultdict(zero)
        spxlong = defaultdict(zero)
        spxlong["spx"] = 1
        mktweights[True] = spxlong
        mktweights[False] = spxlong
        mkt_return = apply_trading_rule(
            dfX, dfrtn, xcol, rtn_col, weights=mktweights
        )
        if not show_plot:
            plt.ioff()

        plt.figure(figsize=(10, 4))

        plt.plot(strat_return["date"], strat_return["totrtn"], label="Long/Short")
        plt.plot(mkt_return["date"], mkt_return["totrtn"], label="SPX")
        plt.title(title)
        plt.legend()
        plt.grid(ls="--", alpha=0.5)
        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if save_plot:
            figname = f"img/trading_rule_plots/{title}.png"
            plt.savefig(figname, dpi=200)

        if show_plot:
            plt.show()

        plt.close()
        plt.ion()
    return apply_trading_rule, plot_trading_rule, pvt_rtn


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
def _(
    apply_trading_rule,
    defaultdict,
    dfXcpi,
    dfreturns,
    plot_trading_rule,
    zero,
):
    _rising_weights = defaultdict(zero)
    _falling_weights = defaultdict(zero)
    _weights = defaultdict(zero)

    _rising_weights["s5tech"] = 1
    _rising_weights["s5enrs"] = 1
    _rising_weights["s5cond"] = -1
    _rising_weights["s5hlth"] = -1

    _falling_weights["s5tech"] = 1
    _falling_weights["s5matr"] = 1
    _falling_weights["s5cons"] = -1
    _falling_weights["s5finl"] = -1


    _weights[True] = _rising_weights
    _weights[False] = _falling_weights

    _xcol = "3M_rising_yoy"
    _rtn_col = "rtn_6M"
    _title = "6M Fwd Returns ~ 3M Rising CPI YoY"
    _xlabel = "Date"
    _ylabel = "Total Return"
    # rtn_cols=[f"rtn_{i+1}M" for i in range(12)]
    strat_rtn = apply_trading_rule(
        dfX=dfXcpi, dfrtn=dfreturns, xcol=_xcol, rtn_col=_rtn_col, weights=_weights
    )
    plot_trading_rule(
        dfX=dfXcpi,
        dfrtn=dfreturns,
        xcol=_xcol,
        rtn_col=_rtn_col,
        weights=_weights,
        # save_plot=True,
        show_plot=True,
        title=_title,
        xlabel=_xlabel,
        ylabel=_ylabel,
    )
    return (strat_rtn,)


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
def _(defaultdict, dfX, dfreturns, plot_trading_rule, zero):
    rising_weights_1 = defaultdict(zero)
    falling_weights_1 = defaultdict(zero)
    weights_1 = defaultdict(zero)
    rising_weights_1["s5tech"] = 1
    rising_weights_1["s5finl"] = 1
    rising_weights_1["s5enrs"] = -1
    rising_weights_1["s5indu"] = -1
    falling_weights_1["s5enrs"] = 1
    falling_weights_1["s5tech"] = 1
    falling_weights_1["s5finl"] = -1
    falling_weights_1["s5cons"] = -1
    weights_1[True] = rising_weights_1
    weights_1[False] = falling_weights_1
    title_1 = "3M Fwd Returns ~ 3M Rising CPI YoY"
    xlabel_1 = "Date"
    ylabel_1 = "Total Return"
    xcol_1 = "3M_rising"
    rtn_col_1 = "rtn_3M"
    plot_trading_rule(
        dfX,
        dfrtn=dfreturns,
        xcol=xcol_1,
        rtn_col=rtn_col_1,
        weights=weights_1,
        save_plot=True,
        show_plot=True,
        title=title_1,
        xlabel=xlabel_1,
        ylabel=ylabel_1,
    )
    return (
        falling_weights_1,
        rising_weights_1,
        rtn_col_1,
        title_1,
        weights_1,
        xcol_1,
        xlabel_1,
        ylabel_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# $\Delta cpi.yoy\$ ~ Mean Return x sector""")
    return


@app.cell
def _(dfX, dfXcpi, pd):
    dfX_1 = pd.merge(
        left=dfX.reset_index(),
        right=dfXcpi.reset_index(),
        left_on="refmonthyear",
        right_on="refmonthyear",
        suffixes=(None, "_y"),
    )
    dfX_1.drop(
        columns=[x for x in dfX_1.columns if x.endswith("_y")],
        axis=1,
        inplace=True,
    )
    dfX_1.set_index(["date", "refmonthyear"], inplace=True)
    return (dfX_1,)


@app.cell
def _(dfrtn, head):
    dfrtn2 = dfrtn.stack(future_stack=True).reset_index()
    head(dfrtn2)
    return (dfrtn2,)


@app.cell
def _(dfXcpi, dfrtn_simple, pd):
    rtn_col_2 = "rtn_1M"
    xcol_2 = "3M_rising_yoy"
    dfrtn_cpi2 = pd.merge(
        dfrtn_simple,
        dfXcpi.reset_index(),
        left_on="refmonthyear",
        right_on="refmonthyear",
        suffixes=("", "_y"),
        how="inner",
    ).drop("date_y", axis="columns")
    return dfrtn_cpi2, rtn_col_2, xcol_2


@app.cell
def _():
    cpi_factors_1 = [
        "3M_rising_yoy",
        "6M_rising_yoy",
        "9M_rising_yoy",
        "12M_rising_yoy",
    ]
    rtn_cols_1 = [f"rtn_{i + 1}M" for i in range(12)]
    return cpi_factors_1, rtn_cols_1


@app.cell
def _(dfrtn_cpi2):
    dfrtn_cpi2
    return


@app.cell
def _(dfrtn_cpi2):
    dfrtn_cpi2[dfrtn_cpi2.ticker == "spx"].to_clipboard()
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
def _(dfrtn_cpi2, factor_barplot):
    xcol_3 = "3M_rising_yoy"
    rtn_col_3 = "rtn_3M"
    title_2 = "3M Fwd Returns ~ 3M Rising CPI YoY"
    ylabel_2 = "Excess 3M Fwd Return"
    xlabel_2 = "Sector Ticker"
    factor_barplot(
        dfrtn_cpi2,
        xcol_3,
        rtn_col_3,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_2,
        xlabel=xlabel_2,
    )
    return rtn_col_3, title_2, xcol_3, xlabel_2, ylabel_2


@app.cell
def _(defaultdict, dfX_1, dfreturns, plot_trading_rule, zero):
    rising_weights_2 = defaultdict(zero)
    falling_weights_2 = defaultdict(zero)
    weights_2 = defaultdict(zero)
    rising_weights_2["s5tech"] = 1
    rising_weights_2["s5hlth"] = 1
    rising_weights_2["s5enrs"] = -1
    rising_weights_2["s5indu"] = -1
    falling_weights_2["s5tech"] = 1
    falling_weights_2["s5finl"] = 1
    falling_weights_2["s5hlth"] = -1
    falling_weights_2["s5cons"] = -1
    weights_2[True] = rising_weights_2
    weights_2[False] = falling_weights_2
    xcol_4 = "3M_rising_yoy"
    rtn_col_4 = "rtn_3M"
    plot_trading_rule(
        dfX_1,
        dfrtn=dfreturns,
        xcol=xcol_4,
        rtn_col=rtn_col_4,
        weights=weights_2,
        save_plot=True,
        show_plot=True,
    )
    return falling_weights_2, rising_weights_2, rtn_col_4, weights_2, xcol_4


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
def _(
    defaultdict,
    dfX_1,
    dfreturns,
    plot_trading_rule,
    rtn_col_4,
    xcol_4,
    zero,
):
    rising_weights_3 = defaultdict(zero)
    falling_weights_3 = defaultdict(zero)
    weights_3 = defaultdict(zero)
    rising_weights_3["s5tech"] = 1
    rising_weights_3["s5hlth"] = 1
    rising_weights_3["s5cond"] = -1
    rising_weights_3["s5indu"] = -1
    falling_weights_3["s5tech"] = 1
    falling_weights_3["s5cond"] = 1
    falling_weights_3["s5hlth"] = -1
    falling_weights_3["s5cons"] = -1
    weights_3[True] = rising_weights_3
    weights_3[False] = falling_weights_3
    title_3 = "3M Fwd Returns ~ 6M Rising CPI YoY"
    xlabel_3 = "Date"
    ylabel_3 = "Total Return"
    plot_trading_rule(
        dfX_1,
        dfrtn=dfreturns,
        xcol=xcol_4,
        rtn_col=rtn_col_4,
        weights=weights_3,
        save_plot=True,
        show_plot=True,
        title=title_3,
        xlabel=xlabel_3,
        ylabel=ylabel_3,
    )
    return (
        falling_weights_3,
        rising_weights_3,
        title_3,
        weights_3,
        xlabel_3,
        ylabel_3,
    )


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
    xcol_5 = "6M_rising_yoy"
    rtn_col_5 = "rtn_6M"
    title_4 = "6M Fwd Returns ~ 6M Rising CPI YoY"
    ylabel_4 = "Excess 6M Fwd Return"
    xlabel_4 = "Sector Ticker"
    plt.rcParams["figure.figsize"] = (6, 3)
    factor_barplot(
        dfrtn_cpi2,
        xcol_5,
        rtn_col_5,
        show_plot=True,
        save_plot=False,
        figsize=(6, 3),
        ylabel=ylabel_4,
        xlabel=xlabel_4,
    )
    return rtn_col_5, title_4, xcol_5, xlabel_4, ylabel_4


@app.cell
def _(
    defaultdict,
    dfX_1,
    dfreturns,
    plot_trading_rule,
    rtn_col_5,
    xcol_5,
    zero,
):
    rising_weights_4 = defaultdict(zero)
    falling_weights_4 = defaultdict(zero)
    weights_4 = defaultdict(zero)
    rising_weights_4["s5tech"] = 1
    rising_weights_4["s5enrs"] = 1
    rising_weights_4["s5finl"] = -1
    rising_weights_4["s5cond"] = -1
    falling_weights_4["s5tech"] = 1
    falling_weights_4["s5finl"] = 1
    falling_weights_4["s5hlth"] = -1
    falling_weights_4["s5cons"] = -1
    weights_4[True] = rising_weights_4
    weights_4[False] = falling_weights_4
    plot_trading_rule(
        dfX_1,
        dfrtn=dfreturns,
        xcol=xcol_5,
        rtn_col=rtn_col_5,
        weights=weights_4,
        save_plot=True,
        show_plot=True,
    )
    return falling_weights_4, rising_weights_4, weights_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Factor barplots pre 2013""")
    return


@app.cell
def _():
    from datetime import date
    return (date,)


@app.cell
def _(date, dfrtn_cpi2, pd):
    mask = dfrtn_cpi2["date"] < pd.to_datetime(date(2013, 1, 1))
    return (mask,)


@app.cell
def _(date, dfrtn_cpi2, pd):
    mask2 = dfrtn_cpi2["date"] >= pd.to_datetime(date(2013, 1, 1))
    return (mask2,)


@app.cell
def _(dfrtn_cpi2, factor_barplot, mask):
    xcol_6 = "3M_rising_yoy"
    rtn_col_6 = "rtn_1M"
    title_5 = "{} Fwd Returns ~ 3M Rising CPI YoY".format(
        rtn_col_6.replace("rtn_", "")
    )
    ylabel_5 = "Excess 3M Fwd Return"
    xlabel_5 = "Sector Ticker"
    factor_barplot(
        dfrtn_cpi2[mask],
        xcol_6,
        rtn_col_6,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_5,
        xlabel=xlabel_5,
        title=f"{title_5} - pre 2013",
    )
    return rtn_col_6, title_5, xcol_6, xlabel_5, ylabel_5


@app.cell
def _(
    dfrtn_cpi2,
    factor_barplot,
    mask2,
    rtn_col_6,
    title_5,
    xcol_6,
    xlabel_5,
    ylabel_5,
):
    factor_barplot(
        dfrtn_cpi2[mask2],
        xcol_6,
        rtn_col_6,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_5,
        xlabel=xlabel_5,
        title=f"{title_5} - post 2013",
    )
    return


@app.cell
def _(
    dfrtn_cpi2,
    factor_barplot,
    rtn_col_6,
    title_5,
    xcol_6,
    xlabel_5,
    ylabel_5,
):
    factor_barplot(
        dfrtn_cpi2,
        xcol_6,
        rtn_col_6,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_5,
        xlabel=xlabel_5,
        title=f"{title_5} - 1989 -- 2023",
    )
    return


@app.cell
def _(dfrtn_cpi2, factor_barplot, mask, mask2):
    xcol_7 = "3M_rising_yoy"
    rtn_col_7 = "rtn_2M"
    title_6 = "{} Fwd Returns ~ 3M Rising CPI YoY".format(
        rtn_col_7.replace("rtn_", "")
    )
    ylabel_6 = "Excess 3M Fwd Return"
    xlabel_6 = "Sector Ticker"
    factor_barplot(
        dfrtn_cpi2[mask],
        xcol_7,
        rtn_col_7,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_6,
        xlabel=xlabel_6,
        title=f"{title_6} - pre 2013",
    )
    factor_barplot(
        dfrtn_cpi2[mask2],
        xcol_7,
        rtn_col_7,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_6,
        xlabel=xlabel_6,
        title=f"{title_6} - post 2013",
    )
    return rtn_col_7, title_6, xcol_7, xlabel_6, ylabel_6


@app.cell
def _(dfrtn_cpi2, factor_barplot, mask, mask2):
    xcol_8 = "3M_rising_yoy"
    rtn_col_8 = "rtn_3M"
    title_7 = "{} Fwd Returns ~ 3M Rising CPI YoY".format(
        rtn_col_8.replace("rtn_", "")
    )
    ylabel_7 = "Excess 3M Fwd Return"
    xlabel_7 = "Sector Ticker"
    factor_barplot(
        dfrtn_cpi2[mask],
        xcol_8,
        rtn_col_8,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_7,
        xlabel=xlabel_7,
        title=f"{title_7} - pre 2013",
    )
    factor_barplot(
        dfrtn_cpi2[mask2],
        xcol_8,
        rtn_col_8,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_7,
        xlabel=xlabel_7,
        title=f"{title_7} - post 2013",
    )
    return rtn_col_8, title_7, xcol_8, xlabel_7, ylabel_7


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 6M Rising YoY""")
    return


@app.cell
def _(dfrtn_cpi2, factor_barplot, mask, mask2):
    xcol_9 = "6M_rising_yoy"
    rtn_col_9 = "rtn_1M"
    title_8 = "{} Fwd Returns ~ {} Rising CPI YoY".format(
        rtn_col_9.replace("rtn_", ""), xcol_9.replace("_rising_yoy", "")
    )
    ylabel_8 = "Excess {} Fwd Return".format(rtn_col_9.replace("rtn_", ""))
    xlabel_8 = "Sector Ticker"
    factor_barplot(
        dfrtn_cpi2[mask],
        xcol_9,
        rtn_col_9,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_8,
        xlabel=xlabel_8,
        title=f"{title_8} - pre 2013",
    )
    factor_barplot(
        dfrtn_cpi2[mask2],
        xcol_9,
        rtn_col_9,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_8,
        xlabel=xlabel_8,
        title=f"{title_8} - post 2013",
    )
    return rtn_col_9, title_8, xcol_9, xlabel_8, ylabel_8


@app.cell
def _(dfrtn_cpi2, factor_barplot, mask, mask2):
    xcol_10 = "6M_rising_yoy"
    rtn_col_10 = "rtn_2M"
    title_9 = "{} Fwd Returns ~ {} Rising CPI YoY".format(
        rtn_col_10.replace("rtn_", ""), xcol_10.replace("_rising_yoy", "")
    )
    ylabel_9 = "Excess {} Fwd Return".format(rtn_col_10.replace("rtn_", ""))
    xlabel_9 = "Sector Ticker"
    factor_barplot(
        dfrtn_cpi2[mask],
        xcol_10,
        rtn_col_10,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_9,
        xlabel=xlabel_9,
        title=f"{title_9} - pre 2013",
    )
    factor_barplot(
        dfrtn_cpi2[mask2],
        xcol_10,
        rtn_col_10,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_9,
        xlabel=xlabel_9,
        title=f"{title_9} - post 2013",
    )
    return rtn_col_10, title_9, xcol_10, xlabel_9, ylabel_9


@app.cell
def _(dfrtn_cpi2, factor_barplot, mask, mask2):
    xcol_11 = "6M_rising_yoy"
    rtn_col_11 = "rtn_3M"
    title_10 = "{} Fwd Returns ~ {} Rising CPI YoY".format(
        rtn_col_11.replace("rtn_", ""), xcol_11.replace("_rising_yoy", "")
    )
    ylabel_10 = "Excess {} Fwd Return".format(rtn_col_11.replace("rtn_", ""))
    xlabel_10 = "Sector Ticker"
    factor_barplot(
        dfrtn_cpi2[mask],
        xcol_11,
        rtn_col_11,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_10,
        xlabel=xlabel_10,
        title=f"{title_10} - pre 2013",
    )
    factor_barplot(
        dfrtn_cpi2[mask2],
        xcol_11,
        rtn_col_11,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_10,
        xlabel=xlabel_10,
        title=f"{title_10} - post 2013",
    )
    return rtn_col_11, title_10, xcol_11, xlabel_10, ylabel_10


@app.cell
def _(dfrtn_cpi2, factor_barplot, mask, mask2):
    xcol_12 = "6M_rising_yoy"
    rtn_col_12 = "rtn_6M"
    title_11 = "{} Fwd Returns ~ {} Rising CPI YoY".format(
        rtn_col_12.replace("rtn_", ""), xcol_12.replace("_rising_yoy", "")
    )
    ylabel_11 = "Excess {} Fwd Return".format(rtn_col_12.replace("rtn_", ""))
    xlabel_11 = "Sector Ticker"
    factor_barplot(
        dfrtn_cpi2[mask],
        xcol_12,
        rtn_col_12,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_11,
        xlabel=xlabel_11,
        title=f"{title_11} - pre 2013",
    )
    factor_barplot(
        dfrtn_cpi2[mask2],
        xcol_12,
        rtn_col_12,
        show_plot=True,
        save_plot=False,
        figsize=(10, 4),
        ylabel=ylabel_11,
        xlabel=xlabel_11,
        title=f"{title_11} - post 2013",
    )
    return rtn_col_12, title_11, xcol_12, xlabel_11, ylabel_11


@app.cell
def _(dfrtn_cpi2):
    dates = dfrtn_cpi2["date"].unique()
    return (dates,)


@app.cell
def _(dates):
    years = list(set([d.year for d in dates]))
    return (years,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Factor barplots""")
    return


@app.cell
def _():
    sector_map = {
        "s5tech": "Tech",
        "s5hlth": "Healthcare",
        "s5enrs": "Energy",
        "s5indu": "Industrials",
        "s5finl": "Financials",
        "s5cond": "Cons Disc",
        "s5cons": "Cons Staples",
        "s5matr": "Materials",
    }

    cpi_rising_map = {True: "CPI Rising", False: "CPI Falling"}
    return cpi_rising_map, sector_map


@app.cell
def _(dfrtn_cpi2):
    xcol_13 = "6M_rising_yoy"
    rtn_col_13 = "rtn_1M"
    title_12 = "1M Fwd Returns ~ 6M Rising CPI YoY"
    ylabel_12 = "Excess 1M Fwd Return"
    xlabel_12 = "Sector Ticker"
    df = dfrtn_cpi2
    show_plot = True
    save_plot = False
    factor_col = xcol_13
    use_excess_returns = True
    return (
        df,
        factor_col,
        rtn_col_13,
        save_plot,
        show_plot,
        title_12,
        use_excess_returns,
        xcol_13,
        xlabel_12,
        ylabel_12,
    )


@app.cell
def _(
    cpi_rising_map,
    df,
    factor_col,
    fm,
    pd,
    plt,
    rtn_col_13,
    sb,
    sector_map,
    title_12,
    use_excess_returns,
):
    sb.reset_orig()
    width_px = 410
    height_px = 171
    dpi = 240
    width_in = 5.6
    height_in = 2.5
    fm.fontManager.addfont("/Users/lcota/Library/Fonts/Raleway-Regular.ttf")
    fm.fontManager.addfont("/Users/lcota/Library/Fonts/Raleway-Light.ttf")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Raleway"
    plt.rcParams["font.style"] = "normal"
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.titlecolor"] = "#595959"
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.labelcolor"] = "#595959"
    plt.rcParams["xtick.color"] = "#595959"
    plt.rcParams["ytick.color"] = "#595959"
    figsize = (width_in, height_in)
    plt.ioff()
    mean_returns = (
        df[df[factor_col].isna() == False]
        .groupby(["ticker", factor_col], observed=True)[rtn_col_13]
        .agg("mean")
    )
    excess_returns = mean_returns - mean_returns["spx"]
    excess_returns = excess_returns.reset_index()
    excess_returns = excess_returns[excess_returns["ticker"] != "spx"]
    returns = mean_returns
    if use_excess_returns:
        returns = excess_returns
        if title_12 is None:
            title_13 = f"Excess Sector Returns {rtn_col_13} x {factor_col}"
    else:
        returns = mean_returns.reset_index()
        if title_13 is None:
            title_13 = f"Sector Returns {rtn_col_13} x {factor_col}"
    returns = pd.DataFrame(returns)
    returns[rtn_col_13] = returns[rtn_col_13] * 100
    returns[factor_col] = pd.Categorical(
        returns[factor_col], categories=[True, False], ordered=True
    )
    returns.replace({"ticker": sector_map}, inplace=True)
    returns.replace({factor_col: cpi_rising_map}, inplace=True)
    plt.figure(figsize=figsize, dpi=dpi)
    sb.barplot(returns, x="ticker", y=rtn_col_13, hue=factor_col, legend=True)
    ax = plt.gca()
    ax.legend(loc="upper left", ncol=2, fontsize=6)
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel("1M Fwd Return", fontsize=8)
    plt.xlabel("Sector", fontsize=8)
    plt.title(
        "1M Fwd Excess Returns ~ 6M Inflation Factor", fontsize=9, color="#595959"
    )
    for spine in ax.spines.values():
        spine.set_color("#595959")
        spine.set_alpha(0.7)
    plt.show()
    return (
        ax,
        dpi,
        excess_returns,
        figsize,
        height_in,
        height_px,
        mean_returns,
        returns,
        spine,
        title_13,
        width_in,
        width_px,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Factor barchart pre 2013""")
    return


@app.cell
def _(df):
    df["date"].dtypes
    return


@app.cell
def _(
    cpi_rising_map,
    df,
    dpi,
    factor_col,
    figsize,
    pd,
    plt,
    rtn_col_13,
    sb,
    sector_map,
    title_13,
    use_excess_returns,
):
    mask_1 = df["date"] < pd.Timestamp(year=2013, month=1, day=1)
    mean_returns_1 = (
        df[mask_1][df[factor_col].isna() == False]
        .groupby(["ticker", factor_col], observed=True)[rtn_col_13]
        .agg("mean")
    )
    excess_returns_1 = mean_returns_1 - mean_returns_1["spx"]
    excess_returns_1 = excess_returns_1.reset_index()
    excess_returns_1 = excess_returns_1[excess_returns_1["ticker"] != "spx"]
    returns_1 = mean_returns_1
    if use_excess_returns:
        returns_1 = excess_returns_1
        if title_13 is None:
            title_14 = f"Excess Sector Returns {rtn_col_13} x {factor_col}"
    else:
        returns_1 = mean_returns_1.reset_index()
        if title_14 is None:
            title_14 = f"Sector Returns {rtn_col_13} x {factor_col}"
    returns_1 = pd.DataFrame(returns_1)
    returns_1[rtn_col_13] = returns_1[rtn_col_13] * 100
    returns_1[factor_col] = pd.Categorical(
        returns_1[factor_col], categories=[True, False], ordered=True
    )
    returns_1.replace({"ticker": sector_map}, inplace=True)
    returns_1.replace({factor_col: cpi_rising_map}, inplace=True)
    plt.figure(figsize=figsize, dpi=dpi)
    sb.barplot(returns_1, x="ticker", y=rtn_col_13, hue=factor_col, legend=True)
    ax_1 = plt.gca()
    ax_1.legend(loc="upper left", ncol=2, fontsize=6)
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel("1M Fwd Return", fontsize=8)
    plt.xlabel("Sector", fontsize=8)
    plt.title(
        "1M Fwd Excess Returns ~ 6M Inflation Factor\n(pre 2013)",
        fontsize=9,
        color="#595959",
    )
    for spine_1 in ax_1.spines.values():
        spine_1.set_color("#595959")
        spine_1.set_alpha(0.7)
    plt.show()
    return (
        ax_1,
        excess_returns_1,
        mask_1,
        mean_returns_1,
        returns_1,
        spine_1,
        title_14,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Factor barchart post 2013""")
    return


@app.cell
def _(
    cpi_rising_map,
    df,
    dpi,
    factor_col,
    figsize,
    pd,
    plt,
    rtn_col_13,
    sb,
    sector_map,
    title_14,
    use_excess_returns,
):
    mask_2 = df["date"] > pd.Timestamp(year=2013, month=1, day=1)
    mean_returns_2 = (
        df[mask_2][df[factor_col].isna() == False]
        .groupby(["ticker", factor_col], observed=True)[rtn_col_13]
        .agg("mean")
    )
    excess_returns_2 = mean_returns_2 - mean_returns_2["spx"]
    excess_returns_2 = excess_returns_2.reset_index()
    excess_returns_2 = excess_returns_2[excess_returns_2["ticker"] != "spx"]
    returns_2 = mean_returns_2
    if use_excess_returns:
        returns_2 = excess_returns_2
        if title_14 is None:
            title_15 = f"Excess Sector Returns {rtn_col_13} x {factor_col}"
    else:
        returns_2 = mean_returns_2.reset_index()
        if title_15 is None:
            title_15 = f"Sector Returns {rtn_col_13} x {factor_col}"
    returns_2 = pd.DataFrame(returns_2)
    returns_2[rtn_col_13] = returns_2[rtn_col_13] * 100
    returns_2[factor_col] = pd.Categorical(
        returns_2[factor_col], categories=[True, False], ordered=True
    )
    returns_2.replace({"ticker": sector_map}, inplace=True)
    returns_2.replace({factor_col: cpi_rising_map}, inplace=True)
    plt.figure(figsize=figsize, dpi=dpi)
    sb.barplot(returns_2, x="ticker", y=rtn_col_13, hue=factor_col, legend=True)
    ax_2 = plt.gca()
    ax_2.legend(loc="upper left", ncol=2, fontsize=6)
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel("1M Fwd Return", fontsize=8)
    plt.xlabel("Sector", fontsize=8)
    plt.title(
        "1M Fwd Excess Returns ~ 6M Inflation Factor\n(post 2013)",
        fontsize=9,
        color="#595959",
    )
    for spine_2 in ax_2.spines.values():
        spine_2.set_color("#595959")
        spine_2.set_alpha(0.7)
    plt.show()
    return (
        ax_2,
        excess_returns_2,
        mask_2,
        mean_returns_2,
        returns_2,
        spine_2,
        title_15,
    )


app._unparsable_cell(
    r"""
    ?sb.barplot
    """,
    name="_"
)


@app.cell
def _(excess_returns_2):
    excess_returns_2.to_clipboard()
    return


@app.cell
def _(factor_col, returns_2, rtn_col_13):
    rising_tickers = returns_2[returns_2[factor_col] == True]["ticker"].values
    rising_returns = returns_2[returns_2[factor_col] == True][rtn_col_13].values
    falling_tickers = returns_2[returns_2[factor_col] == False]["ticker"].values
    falling_returns = returns_2[returns_2[factor_col] == False][rtn_col_13].values
    return falling_returns, falling_tickers, rising_returns, rising_tickers


@app.cell
def _(
    dpi,
    falling_returns,
    falling_tickers,
    figsize,
    plt,
    rising_returns,
    rising_tickers,
):
    (fig, ax_3) = plt.subplots(figsize=figsize, dpi=dpi)
    ax_3.bar(rising_tickers, rising_returns, color="blue", alpha=0.75)
    ax_3.bar(falling_tickers, falling_returns, color="orange", alpha=0.75)
    return ax_3, fig


@app.cell
def _(factor_col, returns_2):
    returns_2[returns_2[factor_col] == True].to_clipboard()
    return


@app.cell
def _(factor_col, returns_2):
    returns_2[returns_2[factor_col] == False].to_clipboard()
    return


@app.cell
def _(
    factor_col,
    plt,
    rtn_col_13,
    save_plot,
    show_plot,
    title_15,
    use_excess_returns,
    xlabel_12,
    ylabel_12,
):
    plt.grid(ls="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.title(title_15)
    if xlabel_12 is not None:
        plt.xlabel(xlabel_12)
    if ylabel_12 is not None:
        plt.ylabel(ylabel_12)
    figname = f"img/excess_return_{factor_col}_{rtn_col_13}.png"
    if use_excess_returns == False:
        figname = f"img/total_return_{factor_col}_{rtn_col_13}.png"
    if save_plot:
        plt.savefig(figname)
    if show_plot:
        plt.show()
    return (figname,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
