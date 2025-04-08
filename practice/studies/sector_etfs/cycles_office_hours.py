import pandas as pd
import seaborn as sb 
from seaborn import lineplot, boxplot, scatterplot, heatmap


df = pd.read_parquet("workspace/dfreturns_with_factors.pq")
sector_tickers = [x for x in df["ticker"].unique() if x != "spx"]
market_sym = "spx"

