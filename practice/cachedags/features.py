import pandas as pd
import numpy as np
from cache import cacheable

def returns() -> pd.DataFrame:
    df = pd.read_csv("~/data/bondsentiment/daily_returns_sp1500.csv", header=True)
    return df

def sector_returns(returns: pd.DataFrame, sector: str)  -> pd.DataFrame:
     sector_returns = returns['SECTOR' == sector].copy()
     agg_returns = returns[['TICKER', 'DATE', 'RETURN', 'SECTOR']].groupby(['DATE', 'SECTOR']).agg()({"SECTOR_RETURN" : 'mean'})
     return agg_returns


def log_returns(sector_returns: pd.DataFrame) -> pd.DataFrame:
    return sector_returns.apply(lambda x: np.log(x + 1))


def add(x: int | float, y: int | float) -> int | float:
    return x + y


def mult(x: int | float, y: int | float) -> int | float:
    return x * y

def rand(n: int) -> float:
    return np.random.normal(n)

@cacheable("cachedir/avg_3wk_spend")
def avg_3wk_spend(spend: pd.Series) -> pd.Series:
    """Rolling 3 week average spend."""
    return spend.rolling(3).mean()

@cacheable("cachedir/acquisition_cost")
def acquisition_cost(avg_3wk_spend: pd.Series, signups: pd.Series) -> pd.Series:
    """The cost per signup in relation to a rolling average of spend."""
    return avg_3wk_spend / signups

def spend_mean(spend: pd.Series) -> float:
    """Shows function creating a scalar. In this case it computes the mean of the entire column."""
    return spend.mean()

def spend_zero_mean(spend: pd.Series, spend_mean: float) -> pd.Series:
    """Shows function that takes a scalar. In this case to zero mean spend."""
    return spend - spend_mean

def spend_std_dev(spend: pd.Series) -> float:
    """Function that computes the standard deviation of the spend column."""
    return spend.std()

def spend_zero_mean_unit_variance(spend_zero_mean: pd.Series, spend_std_dev: float) -> pd.Series:
    """Function showing one way to make spend have zero mean and unit variance."""
    return spend_zero_mean / spend_std_dev













